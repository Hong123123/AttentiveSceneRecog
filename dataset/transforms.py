from torchvision import  transforms as trans
import torch

# @preprocess follows the RedNet pipeline:
# https://paperswithcode.com/paper/rednet-residual-encoder-decoder-network-for
#
# RedNet_data.RandomScale((1.0, 1.4)),
# RedNet_data.RandomHSV((0.9, 1.1), (0.9, 1.1), (25, 25)),
# RedNet_data.RandomCrop(image_h, image_w),
# RedNet_data.RandomFlip(),
# RedNet_data.ToTensor(),
# RedNet_data.Normalize()]

# @or follows the placesNet pipeline:
# https://github.com/CSAILVision/places365/blob/master/run_placesCNN_basic.py


# class ForceToFloat():
#     def __call__(self, x):
#         return x.float()


def force_to_float(x):
    return x.float()


rgb_trans = trans.Compose([
        trans.Resize((256,256)),
        trans.RandomCrop(224),
        trans.RandomHorizontalFlip(),
        trans.RandomVerticalFlip(),
        trans.ToTensor(),
        trans.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

depth_trans = trans.Compose([
        trans.Resize((256,256)),
        trans.RandomCrop(224),
        trans.RandomHorizontalFlip(),
        trans.RandomVerticalFlip(),
        trans.ToTensor(),
        force_to_float,
        trans.Normalize([19050.0], [9650.0])
    ])

train_transform = [
    rgb_trans,
    depth_trans,
    lambda x: torch.tensor(x, dtype=torch.long)
]

train_transform_hha = [
    rgb_trans,
    rgb_trans,
    lambda x: torch.tensor(x, dtype=torch.long)
]
