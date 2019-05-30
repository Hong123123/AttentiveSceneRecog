import torch
from torch import nn
from models.Attention import AttentionV1
import numpy as np
# places ref: https://github.com/CSAILVision/places365/blob/master/run_placesCNN_basic.py


class Squeeze(nn.Module):
    def __init__(self, out_size):
        super(Squeeze, self).__init__()
        self.out_size = out_size

    def forward(self,x):
        return x.view(-1, self.out_size)


class AttenAlex(nn.Module):
    def __init__(self, num_classes, atten_open=True, require_atten_at_nth_conv=[0, 1, 2, 3, 4]):
        super(AttenAlex, self).__init__()
        # input: shape(3,227,227)
        # ouput: shape(365)
        self.out_dim = num_classes
        self.atten_open = atten_open
        self.require_atten = require_atten_at_nth_conv

        self.where_convs = [0,4,8,11,14]  # [0,3,6,8,10]
        self.abs_atten = np.array(self.where_convs)[np.array(self.require_atten)]
        self.Cin = 3
        self.Cout = [64, 192, 384, 256, 256]

        self.r_alex = [self.make_alexnet_without_tail()]  # prevent from registering as Module
        self.d_alex = [self.make_alexnet_without_tail()]  # prevent from registering as Module

        self.r_features = list(self.r_alex[0].children())[0]
        self.r_other = nn.Sequential(*list(self.r_alex[0].children())[1:])

        self.d_features = list(self.d_alex[0].children())[0]
        self.d_other = nn.Sequential(*list(self.d_alex[0].children())[1:])

        self.last = self.make_double_last(self.out_dim)

        # attributes about attention:
        # self.attens: stores all attention modules
        # self.is_atten: whether the i-th layer need attention
        # self.which_atten: which attention module the i-th layer corresponds to
        self.attens, self.is_atten, self.which_atten = self.make_atten()

    def forward_alex(self, x):
        x= self.features(x)
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.last(x)
        return x

    def make_alexnet_without_tail(self):
        return nn.Sequential(
            self.make_features(),
            self.make_avg(),
            self.make_fc()
        )

    def make_atten(self):
        is_atten = [False] * len(self.d_features)
        attens = []
        which_atten = {}
        # in_Cs = [self.Cin, *self.Cout[:-1]]
        Cs = self.Cout
        if self.require_atten is not None:
            for i, c in enumerate(Cs):
                if self.where_convs[i] in self.abs_atten:
                    atten_abs_idx = self.where_convs[i]
                    is_atten[atten_abs_idx] = True
                    attens.append(
                        nn.ModuleList(
                            [
                                AttentionV1(c, freeze_z=not self.atten_open, zero_z=not self.atten_open, batch_norm=True),
                                AttentionV1(c, freeze_z=not self.atten_open, zero_z=not self.atten_open, batch_norm=True)
                            ]
                        )
                    )
                    which_atten[str(atten_abs_idx)]=i
        #             print(attens[self.where_convs[i]][0].Wq.weight.requires_grad)
        # print([i for i,v in enumerate(attens) if v is not False])
        return nn.ModuleList(attens), is_atten, which_atten

    def forward(self, x, d):
        for layer,(f,g) in enumerate(zip(self.r_features, self.d_features)):
            # print('{}-th input dim:{}'.format(layer, x.shape[1]))
            # print(f,g)
            x, d = f(x), g(d)
            # print('{}-th output dim:{}'.format(layer, x.shape[1]))

            # print(self.is_atten)
            is_atten = self.is_atten[layer]
            if is_atten is not False:
                # print('{}-th_atten for {}-th layer'.format(self.which_atten[str(layer)], layer))
                # print('{}th_atten'.format(self.which_atten[str(layer)]))
                fx, fd = self.attens[self.which_atten[str(layer)]].children()
                # print(fx,fd)
                x, d = fx(x,d), fd(d,x)

        for f,g in zip(self.r_other, self.d_other):
            x, d = f(x), g(x)

        v = torch.cat((x,d), dim=1)
        v = self.last(v)
        return v

    def make_features(self):
        C = self.Cout
        Cin = self.Cin
        return nn.Sequential(
            # follow the torchvision implementation rather than the original paper
            # ref: https://pytorch.org/docs/stable/torchvision/models.html?highlight=torchvision%20models

            # conv_weight.shape(out_channels, in_channels // groups, *kernel_size)
            nn.Conv2d(Cin, C[0], kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(C[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # padding=0, dilation=1, ceil_mode=False),
            # nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75),

            nn.Conv2d(C[0], C[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(C[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(C[1], C[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C[2]),
            nn.ReLU(inplace=True),

            nn.Conv2d(C[2], C[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C[3]),
            nn.ReLU(inplace=True),

            nn.Conv2d(C[3], C[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # padding=0, dilation=1, ceil_mode=False)
        )

    @staticmethod
    def make_avg():
        return nn.AdaptiveAvgPool2d(output_size=(6,6))

    @staticmethod
    def make_fc():
        return nn.Sequential(
            Squeeze(256*6*6),
            nn.Dropout(p=0.5),
            nn.Linear(9216, 4096),  # 9126 = 256*6*6 shape(in_dim, out_dim) but its weight.shape(out_dim, in_dim)
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def make_last(out_dim):
        return nn.Linear(4096, out_dim)

    @staticmethod
    def make_double_last(out_dim):
        return nn.Linear(4096*2, out_dim)


if __name__ == '__main__':
    from dataset.sunrgbd_dataset import SunRgbdDataset
    from dataset.transforms import train_transform_hha as tr_hha
    from torch.utils.data import DataLoader
    import config
    device = torch.device('cuda:1')

    alex = AttenAlex(45)

    ckpt = torch.load(config.places_alex, map_location=device)

    # load weight manually
    conv_idxmap=alex.where_convs
    for k,(key,value) in enumerate(ckpt['state_dict'].items()):
        # key = str.replace(key, 'module.', '')  # preprocssing]
        if key.find('features') > -1:
            # key = str.replace(key, 'features', 'r_features')  # transfering Places2Net features
            if k < 10:
                if k % 2 == 0:
                    key = 'r_features.{}.weight'.format(conv_idxmap[k//2])
                else:
                    key = 'r_features.{}.bias'.format(conv_idxmap[k//2])
        else:
            target = 'classifier'  # transfering Places2Net classifiers
            sp = key.split('.')
            assert len(sp) == 3
            if sp[0] == target:
                key = 'r_other.1.{}.{}'.format(int(sp[1])+1, sp[2])
        if key in alex.state_dict().keys():
            print('copied: {}'.format(key))
            alex.state_dict()[key].copy_(value)

    # model = torchvision.models.__dict__['alexnet'](num_classes=45)
    # alex.load_state_dict(state_dict)

    sunset = SunRgbdDataset(config.sunrgbd_root, config.sunrgbd_label_dict_dir, hha_mode=True, transform=tr_hha)
    loader = DataLoader(sunset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
    sample = next(iter(loader))

    print(alex.eval())
    print(sunset.cls_dict.keys())
    print(sample)

    o = alex(sample['rgb'], sample['depth'])
    criterion = nn.CrossEntropyLoss()
    criterion(o, sample['label']).backward()
    predict = np.array(o.detach()).argmax()
    print(predict)
    print(list(sunset.cls_dict.keys())[predict])
    print('frequency: ',sunset.cls_dict[list(sunset.cls_dict.keys())[predict]]['frequency'])
    pass

# for param in alex.attens.parameters():
#     param.requires_grad = False

# if not torch.sum(torch.abs(al.attens[0][0].Wz.weight.flatten())):
#     print('yes')
# else:
#     print('no')