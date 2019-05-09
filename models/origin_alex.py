import torch
import torchvision
from torch import nn
import config

# model = torchvision.models.__dict__['alexnet'](num_classes=365)
# model.load_state_dict(state_dict)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DAlex(nn.Module):
    def __init__(self, cls_num, pretrain_dir=''):
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

        self.last_linear = nn.Linear(4096*2, self.cls_num)

    def forward(self, xr, xd):
        xr = self.rgb_alex(xr)
        xd = self.d_alex(xd)
        concatenated = torch.cat((xr, xd), 1)
        y = self.last_linear(concatenated)
        return y


if __name__ == '__main__':
    import config
    from dataset.transforms import train_transform
    from dataset.nyud2_dataset import NYUD2Dataset
    dalex = DAlex(10, pretrain_dir=config.places_alex).cpu()
    dataset = NYUD2Dataset(config.nyud2_dir, phase='val', transform=train_transform)
    one_sample = dataset[0]
    r = one_sample['rgb'].cpu()
    xi = torch.reshape(r, (1, *r.shape))
    dalex(xi)
    print('oops')
