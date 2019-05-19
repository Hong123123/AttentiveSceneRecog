import torch
import torchvision
from torch import nn
from models.Attention import AttentionV2, MultiHead
import config

# model = torchvision.models.__dict__['alexnet'](num_classes=365)
# model.load_state_dict(state_dict)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DAlex(nn.Module):
    def __init__(self, cls_num, pretrain_dir='', freeze_front=False):
        super(DAlex, self).__init__()
        self.cls_num = cls_num
        self.rgb_alex_ = [torchvision.models.__dict__['alexnet'](num_classes=self.cls_num)]
        self.d_alex_ = [torchvision.models.__dict__['alexnet'](num_classes=self.cls_num)]
        self.rgb_alex_[0].classifier[6] = Identity()  # I mean delete
        self.d_alex_[0].classifier[6] = Identity()

        if pretrain_dir:
            state_dict = torch.load(pretrain_dir)['state_dict']  # , map_location=device)
            state_dict = {k.replace('.module', ''):v for k,v in state_dict.items()}
            state_dict.pop('classifier.6.weight')
            state_dict.pop('classifier.6.bias')
            alex = self.rgb_alex_[0]
            for k, v in state_dict.items():
                if k in alex.state_dict().keys():
                    self.rgb_alex_[0].state_dict()[k].copy_(v)
                    self.d_alex_[0].state_dict()[k].copy_(v)

        self.rgb_alex = self.rgb_alex_[0]
        self.d_alex = self.d_alex_[0]

        if freeze_front:
            for m in (self.rgb_alex, self.d_alex):
                for param in m.parameters():
                    param.requires_grad=False
                print('-'*5, 'freezing', '-'*5, m)

        self.last_linear = nn.Linear(4096*2, self.cls_num)

    def forward(self, xr, xd):
        xr = self.rgb_alex(xr)
        xd = self.d_alex(xd)
        concatenated = torch.cat((xr, xd), 1)
        y = self.last_linear(concatenated)
        return y


class DoubleModule(nn.Module):
    def __init__(self, net1, net2):
        super(DoubleModule, self).__init__()
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


class AttenDAlex(nn.Module):
    def __init__(self, cls_num, add_atten=(1, 4),
                 pretrain_dir='', baseline_dir='', freeze_front=False, raw_atten_no_relu_dir='',
                 atten_type='raw', zero_z=True, freeze_z=True):
        super(AttenDAlex, self).__init__()
        self.cls_num = cls_num
        self.zero_z = zero_z
        self.freeze_z = freeze_z

        self.rgb_alex_ = [torchvision.models.__dict__['alexnet'](num_classes=self.cls_num)]
        self.d_alex_ = [torchvision.models.__dict__['alexnet'](num_classes=self.cls_num)]
        self.rgb_alex_[0].classifier[6] = Identity()  # I mean delete
        self.d_alex_[0].classifier[6] = Identity()
        self.last_linear_ = [nn.Linear(4096 * 2, self.cls_num)]
        self.AttenBase = AttentionV2 if atten_type=='raw' else MultiHead

        if pretrain_dir:  # load 1
            state_dict = torch.load(pretrain_dir)['state_dict']  # , map_location=device)
            state_dict = {k.replace('.module', ''): v for k, v in state_dict.items()}
            state_dict.pop('classifier.6.weight')
            state_dict.pop('classifier.6.bias')
            alex = self.rgb_alex_[0]
            for k, v in state_dict.items():
                if k in alex.state_dict().keys():
                    self.rgb_alex_[0].state_dict()[k].copy_(v)
                    self.d_alex_[0].state_dict()[k].copy_(v)
        elif baseline_dir:  # load 2
            state_dict = torch.load(baseline_dir)['state_dict']
            rgbkeys=[k for k in state_dict.keys() if 'rgb_alex.' in k]
            dkeys=[k for k in state_dict.keys() if 'd_alex.' in k]

            for rkey, dkey in zip(rgbkeys, dkeys):
                self.rgb_alex_[0].state_dict()[rkey.replace('rgb_alex.', '')].copy_(state_dict[rkey])
                self.d_alex_[0].state_dict()[dkey.replace('d_alex.', '')].copy_(state_dict[dkey])

            self.last_linear_[0].state_dict()['weight'].copy_(state_dict['last_linear.weight'])
            self.last_linear_[0].state_dict()['bias'].copy_(state_dict['last_linear.bias'])

        if freeze_front:
            for m in (self.rgb_alex_[0], self.d_alex_[0]):
                for param in m.parameters():
                    param.requires_grad = False
                print('-' * 5, 'freezing', '-' * 5, m)

        # >>> unfold alex
        self.features = nn.Sequential(
            *(
                DoubleModule(rlayer, dlayer)
                for rlayer, dlayer in zip(self.rgb_alex_[0].features,self.d_alex_[0].features)
            )
        )
        self.avgpool = DoubleModule(self.rgb_alex_[0].avgpool, self.d_alex_[0].avgpool)
        self.classifier = nn.Sequential(
            *(
                DoubleModule(rlayer, dlayer)
                for rlayer, dlayer in zip(self.rgb_alex_[0].classifier, self.d_alex_[0].classifier)
            )
        )
        self.last_linear = self.last_linear_[0]
        # <<<

        self.alex_conv_idx = [0,3,6,8,10]
        self.add_atten = add_atten
        self.add_atten_after_relu = 0

        if self.add_atten:
            for conv_idx in self.add_atten:
                layer_idx = self.alex_conv_idx[conv_idx] + self.add_atten_after_relu
                doublelayer = self.features[layer_idx]
                C = doublelayer.net1.out_channels
                double_atten = DoubleAttention(
                    self.AttenBase(C,batch_norm=True,zero_z=self.zero_z,freeze_z=self.freeze_z),
                    self.AttenBase(C,batch_norm=True,zero_z=self.zero_z,freeze_z=self.freeze_z)
                )
                self.features[layer_idx] = nn.Sequential(
                    doublelayer,
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


class AttenDAlexRelu(AttenDAlex):
    def __init__(self, cls_num, add_atten=(1, 4),
                 pretrain_dir='', baseline_dir='', freeze_front=False, raw_atten_no_relu_dir='',
                 atten_type='raw', zero_z=True, freeze_z=True):
        super(AttenDAlexRelu, self).__init__(cls_num, add_atten=add_atten,
                 pretrain_dir=pretrain_dir, baseline_dir=baseline_dir, freeze_front=freeze_front,
                 atten_type=atten_type, zero_z=zero_z, freeze_z=freeze_z)

        if raw_atten_no_relu_dir:
            state_dict_without_relu = torch.load(raw_atten_no_relu_dir)['state_dict']
            self.load_state_dict(state_dict_without_relu)

        if self.add_atten:
            for conv_idx in self.add_atten:
                layer_idx = self.alex_conv_idx[conv_idx] + self.add_atten_after_relu
                doublelayer = self.features[layer_idx]
                double_relu = DoubleModule(
                    nn.ReLU(inplace=False),
                    nn.ReLU(inplace=False)
                )
                self.features[layer_idx] = nn.Sequential(
                    doublelayer,
                    double_relu
                )


# copy codes from AttenDAlex
class ResidualAttenDAlex(nn.Module):
    def __init__(self, cls_num, add_atten=(1, 4),
                 pretrain_dir='', baseline_dir='', freeze_front=False, zero_z=True, freeze_z=True):
        super(ResidualAttenDAlex, self).__init__()
        self.cls_num = cls_num
        self.zero_z = zero_z
        self.freeze_z = freeze_z

        self.rgb_alex_ = [torchvision.models.__dict__['alexnet'](num_classes=self.cls_num)]
        self.d_alex_ = [torchvision.models.__dict__['alexnet'](num_classes=self.cls_num)]
        self.rgb_alex_[0].classifier[6] = Identity()  # I mean delete
        self.d_alex_[0].classifier[6] = Identity()
        self.last_linear_ = [nn.Linear(4096 * 2, self.cls_num)]
        # self.AttenBase = AttentionV2 if atten_type=='raw' else MultiHead

        if pretrain_dir:  # load 1
            state_dict = torch.load(pretrain_dir)['state_dict']  # , map_location=device)
            state_dict = {k.replace('.module', ''): v for k, v in state_dict.items()}
            state_dict.pop('classifier.6.weight')
            state_dict.pop('classifier.6.bias')
            alex = self.rgb_alex_[0]
            for k, v in state_dict.items():
                if k in alex.state_dict().keys():
                    self.rgb_alex_[0].state_dict()[k].copy_(v)
                    self.d_alex_[0].state_dict()[k].copy_(v)
        elif baseline_dir:  # load 2
            state_dict = torch.load(baseline_dir)['state_dict']
            rgbkeys=[k for k in state_dict.keys() if 'rgb_alex.' in k]
            dkeys=[k for k in state_dict.keys() if 'd_alex.' in k]

            for rkey, dkey in zip(rgbkeys, dkeys):
                self.rgb_alex_[0].state_dict()[rkey.replace('rgb_alex.', '')].copy_(state_dict[rkey])
                self.d_alex_[0].state_dict()[dkey.replace('d_alex.', '')].copy_(state_dict[dkey])

            self.last_linear_[0].state_dict()['weight'].copy_(state_dict['last_linear.weight'])
            self.last_linear_[0].state_dict()['bias'].copy_(state_dict['last_linear.bias'])

        if freeze_front:
            for m in (self.rgb_alex_[0], self.d_alex_[0]):
                for param in m.parameters():
                    param.requires_grad = False
                print('-' * 5, 'freezing', '-' * 5, m)

        # >>> unfold alex
        self.features = nn.Sequential(
            *(
                DoubleModule(rlayer, dlayer)
                for rlayer, dlayer in zip(self.rgb_alex_[0].features,self.d_alex_[0].features)
            )
        )
        self.avgpool = DoubleModule(self.rgb_alex_[0].avgpool, self.d_alex_[0].avgpool)
        self.classifier = nn.Sequential(
            *(
                DoubleModule(rlayer, dlayer)
                for rlayer, dlayer in zip(self.rgb_alex_[0].classifier, self.d_alex_[0].classifier)
            )
        )
        self.last_linear = self.last_linear_[0]
        # <<<

        self.alex_conv_idx = [0,3,6,8,10]
        self.add_atten = add_atten
        self.add_atten_after_relu = 0

        if self.add_atten:
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

    def forward(self, xr, xd):
        X = (xr, xd)
        X = self.features(X)
        X = self.avgpool(X)
        X = (x.view(-1, 256*6*6) for x in X)
        X = self.classifier(X)
        concatenated = torch.cat(X, 1)
        logit = self.last_linear(concatenated)
        return logit


if __name__ == '__main__':
    import config
    from dataset.transforms import train_transform
    from dataset.nyud2_dataset import NYUD2Dataset

    dalex = DAlex(10, pretrain_dir=config.places_alex).cpu()
    aalex = AttenDAlex(10, pretrain_dir=config.places_alex).cpu()
    maalex = AttenDAlex(10, atten_type='multi', pretrain_dir=config.places_alex).cpu()
    raalex = ResidualAttenDAlex(10, pretrain_dir=config.places_alex).cpu()

    dataset = NYUD2Dataset(config.nyud2_dir, phase='val', transform=train_transform)
    one_sample = dataset[0]
    r = one_sample['rgb'].cpu()
    d = one_sample['depth'].cpu()
    r,d = (torch.reshape(v, (1, *v.shape)) for v in (r,d))

    # print(dalex(r, d).shape)
    # print(aalex(r, d).shape)
    # print(maalex(r,d).shape)
    print(raalex(r,d).shape)

    print('oops')
