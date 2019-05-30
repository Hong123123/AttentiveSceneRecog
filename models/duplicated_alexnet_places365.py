import torch
from torch import nn
import torchvision
# places ref: https://github.com/CSAILVision/places365/blob/master/run_placesCNN_basic.py


class AlexnetPlaces365(nn.Module):
    def __init__(self, num_classes):
        super(AlexnetPlaces365, self).__init__()
        # input: shape(3,227,227)
        # ouput: shape(365)
        self.out_dim = num_classes

        self.features = nn.Sequential(
            # follow the torchvision implementation rather than the original paper
            # ref: https://pytorch.org/docs/stable/torchvision/models.html?highlight=torchvision%20models

            # conv_weight.shape(out_channels, in_channels // groups, *kernel_size)
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # padding=0, dilation=1, ceil_mode=False),
            # nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # padding=0, dilation=1, ceil_mode=False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            # fc_weight.shape(out_dim, in_dim)
            nn.Linear(9216, 4096),  # 9126 = 256*6*6
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.avgpool(self.features(x))
        x = x.view(-1, 256*6*6)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    import config
    device = torch.device('cuda:1')
    # ckpt = torch.load(path, map_location=device)
    ckpt = torch.load(config.places_alex, map_location=device)
    state_dict = {str.replace(k, 'module.', ''):v for k,v in ckpt['state_dict'].items()}

    alex = AlexnetPlaces365(365)

    model = torchvision.models.__dict__['alexnet'](num_classes=365)
    model.load_state_dict(state_dict)
    pass
