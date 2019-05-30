import torch
from torch import nn
from models.dynamic_atten_alex import DoubleBranchAlex


tanh = nn.Tanh()


def stanh(x):
    return 1.7159 * tanh(2 * x / 3)


class ThreeInThreeOutModule(nn.Module):
    def __init__(self, net1, net2, net3):
        super(ThreeInThreeOutModule, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.net3 = net3

    def forward(self, x1, x2, x3):
        return self.net1(x1), self.net2(x2), self.net3(x3)


class Instructor(nn.Module):  # si
    def __init__(self, s_channel, g_channel):
        super(Instructor, self).__init__()
        self.Ws = nn.Linear(s_channel, s_channel, bias=True)
        self.Wg = nn.Linear(g_channel, s_channel, bias=False)
        self.s_channel = s_channel
        self.g_channel = g_channel

    def forward(self, si_1, gi):
        si = self.Ws(si_1) + self.Wg(gi)
        si = stanh(si)
        return si


class CrossModalityAttention(nn.Module):  # the gi
    def __init__(self, s_channel=256, c_channel=None, vmid_channel=256, d_channel=256, g_channel=256):
        super(CrossModalityAttention, self).__init__()

        self.s_channel = s_channel
        self.c_channel = c_channel
        self.vmid_channel = vmid_channel

        self.d_channel = d_channel

        self.Wc1 = nn.Linear(c_channel, d_channel, bias=True)
        self.Wc2 = nn.Linear(c_channel, d_channel, bias=True)

        self.WB = nn.Linear(s_channel, vmid_channel, bias=False)
        self.VB1 = nn.Linear(c_channel, vmid_channel, bias=True)
        self.VB2 = nn.Linear(c_channel, vmid_channel, bias=True)

        self.wB = nn.Linear(vmid_channel, 1, bias=True)  # scalar
        self.softmax = nn.Softmax(1)

        self.Ws = nn.Linear(s_channel, g_channel, bias=True)

    def forward(self, c1, c2, s):
        B, C = c1.shape
        v1 = self.wB(stanh(self.WB(s) + self.VB1(c1)))  # Shape(B,1), batched scalar
        v2 = self.wB(stanh(self.WB(s) + self.VB2(c2)))  # Shape(B,1)
        v = torch.cat((v1, v2), 1).view(B, 1, -1)  # Shape(B,1,2)

        d1 = self.Wc1(c1)  # Shape(B, D), D for d_channel
        d2 = self.Wc2(c2)
        d = torch.cat((d1.view(B, 1, -1), d2.view(B, 1, -1)), 1)  # Shape(B,2,D)

        beta = self.softmax(v)
        z = torch.einsum('blj,bjd->bld', beta, d).view(B, -1)  # Shape(B,D)
        g = stanh(self.Ws(s) + z)
        return g, beta


class IntraModalityAttention(nn.Module):  # the c_k,i, Shape(Batch, in_channel)
    def __init__(self, in_channel, s_channel=256, mid_channel=256, atten_topk=None):
        super(IntraModalityAttention, self).__init__()

        self.in_channel = in_channel
        self.s_channel = s_channel
        self.mid_channel = mid_channel
        self.atten_topk = atten_topk

        self.Wq = nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=True)  # Out Shape(B, Cmid, H, W)
        self.Uq = nn.Linear(s_channel, mid_channel, bias=False)  # Out Shape(B, Cmid)
        self.w = nn.Conv2d(mid_channel, in_channel, kernel_size=1, bias=True)  # Out Shape(B, Cout)
        self.softmax_dim2 = nn.Softmax(2)

    def forward(self, x, s):
        B, _, H, W = x.shape  # Shape(B, Cin, H, W)

        sig = stanh(self.Wq(x) + self.Uq(s).view(B, -1, 1, 1))  # Shape(B, Cmid, H, W), broadcast involved
        # torch.einsum('om, bmhw -> bohw', w, sig) + b.view(1, b.shape[0], 1, 1)

        alpha = self.w(sig)  # Shape(B, Cin, H, W)
        alpha = alpha.view(B, self.in_channel, -1)  # Shape(B, Cin, H*W)

        # crop attention
        if self.atten_topk is not None:
            if 0 < self.atten_topk < 1:
                topk_num = int(self.atten_topk*H*W)
                # print(topk_num)
            else:
                topk_num = self.atten_topk
                # print(topk_num)
            topk, _ = torch.topk(alpha, topk_num, dim=2, largest=True, sorted=True)
            top_k_th = topk[:,:,-1::]
            alpha[alpha < top_k_th]=0
        else:
            print('no atten cropped')
        atten = self.softmax_dim2(alpha)

        # weighted dot production over spatial dimension per sample per channel
        # (B,C,1,hw) * (B,C,hw,1) -> (B,C,1,1)
        oprand1 = x.view(B, self.in_channel, 1, -1)
        oprand2 = atten.view(*atten.shape, 1)
        c = torch.einsum('bcij, bcjk -> bcik', oprand1, oprand2).view(B, self.in_channel)
        return c, atten


class TripleInstructorLayer(nn.Module):
    def __init__(
            self, conv_channel, s_channel=4096, g_channel=4096,
            mid_channel=4096, atten_topk=100,
            vmid_channel=4096, d_channel=4096
    ):
        super(TripleInstructorLayer, self).__init__()
        self.s = Instructor(s_channel, g_channel)  # Forward(s,g) -> s

        self.atten1 = IntraModalityAttention(  # Forward(x,s) -> c,atten
            conv_channel, s_channel, mid_channel, atten_topk
        )
        self.atten2 = IntraModalityAttention(  # Forward(x,s) -> c,atten
            conv_channel, s_channel, mid_channel, atten_topk
        )
        self.cross_atten = CrossModalityAttention(  # Forward(c1,c2,s) -> g,beta
            s_channel, conv_channel, vmid_channel, d_channel, g_channel
        )

    def forward(self, s, x1, x2):
        c1, atten1 = self.atten1(x1,s)
        c2, atten2 = self.atten1(x2,s)
        g, beta = self.cross_atten(c1,c2,s)
        s = self.s(s,g)
        return s


class InstructorAlex(nn.Module):
    def __init__(
            self, cls_num, pretrain_dir='', baseline_dir='', freeze_front=False, freeze_features=False,
            add_atten_at_conv=None, after_relu=None, atten_topk=100, s_channel=4096
    ):
        super(InstructorAlex, self).__init__()
        self.dbalex_ = [
            DoubleBranchAlex(cls_num, pretrain_dir=pretrain_dir,
                             freeze_front=freeze_front, freeze_features=freeze_features)
        ]
        if baseline_dir:
            sd = torch.load(baseline_dir)['state_dict']
            self.dbalex_[0].load_state_dict(sd)

        self.conv_map = (0, 3, 6, 8, 10)
        self.add_atten_at_conv = (0, 1, 2, 3, 4) if add_atten_at_conv is None else add_atten_at_conv
        self.after_relu = 1 if after_relu is None else after_relu

        self.atten_indice = (self.conv_map[k] + self.after_relu for k in self.add_atten_at_conv)
        self.fc_features = (9216, 4096, 4096, 10)
        self.dbfc2_channel = 4096 * 2
        self.s_channel = s_channel
        self.atten_topk = atten_topk

        self.rgb_alex = self.dbalex_[0].rgb_alex
        self.d_alex = self.dbalex_[0].d_alex
        self.Ws0 = nn.Linear(self.dbfc2_channel, self.s_channel)
        self.last_linear = nn.Linear(self.s_channel + self.dbfc2_channel, 10)

        self.instructor_list = nn.ModuleList()
        self.rgb_conv_activations = {}  # Key('0', '1', '2', ...)
        self.d_conv_activations = {}
        # self.fc2_activations = {}  # Key('r', 'd')

        def hook_save_value_in_dict(dic, key):
            def hook(model, input, output):
                dic[key] = output.detach()
            return hook

        for ith_atten in self.add_atten_at_conv:
            real_idx = self.conv_map[ith_atten] + self.after_relu

            # add hook
            save_rgb_convs_activation = hook_save_value_in_dict(self.rgb_conv_activations, str(ith_atten))
            save_d_convs_activation = hook_save_value_in_dict(self.d_conv_activations, str(ith_atten))
            self.dbalex_[0].rgb_alex.features[real_idx].register_forward_hook(save_rgb_convs_activation)
            self.dbalex_[0].d_alex.features[real_idx].register_forward_hook(save_d_convs_activation)

            # define instructors
            conv_idx = real_idx - self.after_relu
            conv_channel = self.dbalex_[0].rgb_alex.features[conv_idx].out_channels
            self.instructor_list.append(TripleInstructorLayer(
                conv_channel, atten_topk=self.atten_topk,
                s_channel=s_channel, g_channel=s_channel,
                mid_channel=s_channel,vmid_channel=s_channel, d_channel=s_channel
            ))

    def forward(self, r, d):
        r_fc2 = self.rgb_alex(r)
        d_fc2 = self.d_alex(d)

        s0 = torch.cat((r_fc2, d_fc2), 1)  # Shape(C, 8192)
        s = self.Ws0(s0)

        for k,instructor in enumerate(self.instructor_list):
            x1, x2 = self.rgb_conv_activations[str(k)], self.d_conv_activations[str(k)]
            s = instructor(s, x1, x2)
        sn = s  # Shape(B, S)

        augmented = torch.cat((r_fc2, d_fc2, sn), 1)
        y = self.last_linear(augmented)
        return y


class OnlyInstructorAlex(InstructorAlex):  # bad performance
    def __init__(
            self, cls_num, pretrain_dir='', baseline_dir='', freeze_front=False, freeze_features=False,
            add_atten_at_conv=None, after_relu=None
    ):
        super(OnlyInstructorAlex, self).__init__(
            cls_num,
            pretrain_dir=pretrain_dir,
            baseline_dir=baseline_dir,
            freeze_front=freeze_front,
            freeze_features=freeze_features,
            add_atten_at_conv=add_atten_at_conv, after_relu=after_relu
        )
        self.last_linear = nn.Linear(self.s_channel, 10, bias=True)

    def forward(self, r, d):
        r_fc2 = self.rgb_alex(r)
        d_fc2 = self.d_alex(d)

        s0 = torch.cat((r_fc2, d_fc2), 1)  # Shape(C, 8192)
        s = self.Ws0(s0)

        for k,instructor in enumerate(self.instructor_list):
            x1, x2 = self.rgb_conv_activations[str(k)], self.d_conv_activations[str(k)]
            s = instructor(s, x1, x2)
        sn = s  # Shape(B, S)

        y = self.last_linear(sn)
        return y


if __name__ == '__main__':
    import config
    from dataset.transforms import train_transform
    from dataset.nyud2_dataset import NYUD2Dataset

    ialex = InstructorAlex(10, pretrain_dir=config.places_alex, freeze_features=True).cpu()

    dataset = NYUD2Dataset(config.nyud2_dir, phase='val', transform=train_transform)
    one_sample = dataset[0]
    r = one_sample['rgb'].cpu()
    d = one_sample['depth'].cpu()
    r,d = (torch.reshape(v, (1, *v.shape)) for v in (r,d))

    print(ialex(r,d).shape)
    print('oops')
