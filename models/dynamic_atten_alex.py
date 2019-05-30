import torch
import torchvision
from torch import nn
from models.Attention import AttentionV2, MultiHead
from models.basic_op import Identity
import config


# model = torchvision.models.__dict__['alexnet'](num_classes=365)
# model.load_state_dict(state_dict)

class TwoInTwoOutModule(nn.Module):
    def __init__(self, net1, net2):
        super(TwoInTwoOutModule, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, inputs):
        x1,x2 = inputs
        return self.net1(x1), self.net2(x2)


class DoubleAttention(nn.Module):
    def __init__(self, atten1, atten2):
        super(DoubleAttention, self).__init__()
        self.atten1 = atten1
        self.atten2 = atten2

    def forward(self, inputs):
        xr, xd = inputs
        xr, xd = self.atten1(xr,xd), self.atten2(xd, xr)
        return xr, xd


class DoubleResidualAttention(nn.Module):
    def __init__(self, conv1, conv2, atten1, atten2):
        super(DoubleResidualAttention, self).__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.atten1 = atten1
        self.atten2 = atten2

    def forward(self, inputs):
        xr, xd = inputs
        yr = self.conv1(xr) + self.atten1(xr,xd)
        yd = self.conv2(xd) + self.atten2(xd,xr)
        return yr, yd


def freeze_param(model):
    for param in model.parameters():
        param.requires_grad = False


# baseline
class DoubleBranchAlex(nn.Module):
    def __init__(self, cls_num, pretrain_dir='', freeze_front=False, freeze_features=False):
        super(DoubleBranchAlex, self).__init__()
        self.cls_num = cls_num
        self.rgb_alex_ = [torchvision.models.__dict__['alexnet'](num_classes=self.cls_num)]
        self.d_alex_ = [torchvision.models.__dict__['alexnet'](num_classes=self.cls_num)]
        self.rgb_alex_[0].classifier[6] = Identity()  # I mean delete
        self.d_alex_[0].classifier[6] = Identity()

        self.rgb_alex = self.rgb_alex_[0]
        self.d_alex = self.d_alex_[0]

        self.last_linear = nn.Linear(4096*2, self.cls_num)

        if pretrain_dir:
            self.load_pretrain(self.rgb_alex, pretrain_dir)
            self.load_pretrain(self.d_alex, pretrain_dir)

        if freeze_front:
            freeze_param(self.rgb_alex)
            freeze_param(self.d_alex)

        if freeze_features:
            freeze_param(self.rgb_alex_[0].features)
            freeze_param(self.d_alex_[0].features)

    @staticmethod
    def load_pretrain(alex, pretrain_dir):
        state_dict = torch.load(pretrain_dir)['state_dict']  # , map_location=device)
        state_dict = {k.replace('.module', ''): v for k, v in state_dict.items()}
        state_dict.pop('classifier.6.weight')
        state_dict.pop('classifier.6.bias')
        for k, v in state_dict.items():
            if k in alex.state_dict().keys():
                alex.state_dict()[k].copy_(v)

    @staticmethod
    def load_from_parts(ralex_front, dalex_front, last, baseline_dir):
        state_dict = torch.load(baseline_dir)['state_dict']
        rgbkeys = [k for k in state_dict.keys() if 'rgb_alex.' in k]
        dkeys = [k for k in state_dict.keys() if 'd_alex.' in k]

        for rkey, dkey in zip(rgbkeys, dkeys):
            ralex_front.state_dict()[rkey.replace('rgb_alex.', '')].copy_(state_dict[rkey])
            dalex_front.state_dict()[dkey.replace('d_alex.', '')].copy_(state_dict[dkey])

        last.state_dict()['weight'].copy_(state_dict['last_linear.weight'])
        last.state_dict()['bias'].copy_(state_dict['last_linear.bias'])

    def forward(self, xr, xd):
        xr = self.rgb_alex(xr)
        xd = self.d_alex(xd)
        concatenated = torch.cat((xr, xd), 1)
        y = self.last_linear(concatenated)
        return y


# raw or multi head
# contain inside a DoubleBranchAlex class
class AttenDBAlex(nn.Module):
    def __init__(self, cls_num, pretrain_dir='', freeze_front=False, freeze_features=False,
                 baseline_dir='',
                 add_atten_method=None,
                 add_atten=(1, 4), atten_type='raw', zero_z=True, freeze_z=True, add_atten_after_relu=0):
        super(AttenDBAlex, self).__init__()
        self.atten_type = atten_type
        if not atten_type:
            self.atten_type='raw'
        self.cls_num = cls_num
        self.zero_z = zero_z
        self.freeze_z = freeze_z

        self.dbalex_ = [DoubleBranchAlex(cls_num, pretrain_dir=pretrain_dir,
                                         freeze_front=freeze_front,freeze_features=freeze_features)]

        self.rgb_alex_ = [self.dbalex_[0].rgb_alex]
        self.d_alex_ = [self.dbalex_[0].d_alex]
        self.last_linear_ = [self.dbalex_[0].last_linear]

        self.AttenBase = AttentionV2 if self.atten_type=='raw' else MultiHead

        if baseline_dir:  # load 2
            # DBAlex.load_from_parts(self.rgb_alex_[0], self.d_alex_[0], self.last_linear_[0], baseline_dir)
            state_dict = torch.load(baseline_dir)['state_dict']
            self.dbalex_[0].load_state_dict(state_dict)

        # >>> unfold dbalex_
        self.features = nn.Sequential(
            *(
                TwoInTwoOutModule(rlayer, dlayer)
                for rlayer, dlayer in zip(self.rgb_alex_[0].features,self.d_alex_[0].features)
            )
        )
        self.avgpool = TwoInTwoOutModule(self.rgb_alex_[0].avgpool, self.d_alex_[0].avgpool)
        self.classifier = nn.Sequential(
            *(
                TwoInTwoOutModule(rlayer, dlayer)
                for rlayer, dlayer in zip(self.rgb_alex_[0].classifier, self.d_alex_[0].classifier)
            )
        )
        self.last_linear = self.last_linear_[0]
        # <<<

        self.alex_conv_idx = [0,3,6,8,10]
        self.add_atten = add_atten
        self.add_atten_after_relu = add_atten_after_relu

        if self.add_atten:
            if not add_atten_method:
                add_atten_method = self.add_atten_layer
            add_atten_method()

    def add_atten_layer(self):
        for conv_idx in self.add_atten:
            layer_idx = self.alex_conv_idx[conv_idx] + self.add_atten_after_relu
            doublelayer = self.features[layer_idx - self.add_atten_after_relu]
            C = doublelayer.net1.out_channels
            double_atten = DoubleAttention(
                self.AttenBase(C, batch_norm=True, zero_z=self.zero_z, freeze_z=self.freeze_z),
                self.AttenBase(C, batch_norm=True, zero_z=self.zero_z, freeze_z=self.freeze_z)
            )
            self.features[layer_idx] = nn.Sequential(
                self.features[layer_idx],
                double_atten
            )

    def forward(self, xr, xd):
        X = (xr, xd)
        X = self.features(X)
        X = self.avgpool(X)
        X = (x.view(-1, 256*6*6) for x in X)
        X = self.classifier(X)
        concatenated = torch.cat(X, 1)
        logit = self.last_linear(concatenated)
        return logit


class AttenDBAlexRelu(AttenDBAlex):
    def __init__(self, cls_num, add_atten=(1, 4),
                 pretrain_dir='', baseline_dir='', freeze_front=False, AttenDBAlex_dir='',
                 atten_type='raw', zero_z=True, freeze_z=True):
        super(AttenDBAlexRelu, self).__init__(cls_num, add_atten=add_atten,
                                              pretrain_dir=pretrain_dir, baseline_dir=baseline_dir, freeze_front=freeze_front,
                                              atten_type=atten_type, zero_z=zero_z, freeze_z=freeze_z)

        if AttenDBAlex_dir:
            state_dict_without_relu = torch.load(AttenDBAlex_dir)['state_dict']
            self.load_state_dict(state_dict_without_relu)

        if self.add_atten:
            for conv_idx in self.add_atten:
                layer_idx = self.alex_conv_idx[conv_idx] + self.add_atten_after_relu
                doublelayer = self.features[layer_idx]
                double_relu = TwoInTwoOutModule(
                    nn.ReLU(inplace=False),
                    nn.ReLU(inplace=False)
                )
                self.features[layer_idx] = nn.Sequential(
                    doublelayer,
                    double_relu
                )


class ResidualAttenDBAlex(AttenDBAlex):
    def __init__(self, cls_num, pretrain_dir='', freeze_front=False,
                 baseline_dir='',
                 add_atten=(1, 4), atten_type='raw', zero_z=True, freeze_z=True):

        super(ResidualAttenDBAlex, self).__init__(
            cls_num, pretrain_dir=pretrain_dir, freeze_front=freeze_front,
                 baseline_dir=baseline_dir,
                 add_atten=add_atten, atten_type=atten_type, zero_z=zero_z, freeze_z=freeze_z,
                 add_atten_method=self.add_atten_layer
        )

    def add_atten_layer(self):
        print('enter ReAttention')
        for conv_idx in self.add_atten:
            layer_idx = self.alex_conv_idx[conv_idx] + self.add_atten_after_relu
            doublelayer = self.features[layer_idx]

            # actual replacement:
            conv1 = doublelayer.net1
            conv2 = doublelayer.net2
            Cin = conv1.in_channels
            Cout = conv2.out_channels

            atten1 = AttentionV2(Cin, Cout, sliding_bottleneck=True,
                                 batch_norm=True, residual=False,
                                 zero_z=self.zero_z, freeze_z=self.freeze_z)
            atten2 = AttentionV2(Cin, Cout, sliding_bottleneck=True,
                                 batch_norm=True, residual=False,
                                 zero_z=self.zero_z, freeze_z=self.freeze_z)

            self.features[layer_idx] = DoubleResidualAttention(
                conv1, conv2, atten1, atten2
            )


if __name__ == '__main__':
    import config
    from dataset.transforms import train_transform
    from dataset.nyud2_dataset import NYUD2Dataset

    dbalex = DoubleBranchAlex(10, pretrain_dir=config.places_alex, freeze_features=True).cpu()
    aalex = AttenDBAlex(10,
                        baseline_dir='/home/hong/small_log_atten/nyudv2/freeze_baseline/freeze_only_conv/sgd/2019-05-20_00:05:51/checkpoints/best/val_acc_of_0.6284403669724771_at_epoch:926.ckpt'
                        ).cpu()
    maalex = AttenDBAlex(10, atten_type='raw', pretrain_dir=config.places_alex).cpu()
    raalex = ResidualAttenDBAlex(10, pretrain_dir=config.places_alex).cpu()

    dataset = NYUD2Dataset(config.nyud2_dir, phase='val', transform=train_transform)
    one_sample = dataset[0]
    r = one_sample['rgb'].cpu()
    d = one_sample['depth'].cpu()
    r,d = (torch.reshape(v, (1, *v.shape)) for v in (r,d))

    # print(dbalex_(r, d).shape)
    # print(aalex(r, d).shape)
    print(maalex(r,d).shape)
    print(raalex(r,d).shape)

    print('oops')
