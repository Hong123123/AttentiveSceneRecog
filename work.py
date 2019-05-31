# from dataset.sunrgbd_dataset import SunRgbdDataset
from dataset.nyud2_dataset import NYUD2Dataset
from dataset.transforms import train_transform_hha as tr_hha, test_transform_hha as te_hha
# from models.Atten_alex import AttenAlex
from models.dynamic_atten_alex import DoubleBranchAlex, AttenDBAlex
import config
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from datetime import datetime
from tensorboardX import SummaryWriter
import numpy as np
from utility.train_utils import str2bool
from Worker import Worker
from utility.train_utils import get_parameters


# hyperparameters
args = get_parameters()

log_root = args.log_root  # config.log_root  # '/mnt/old_hexin/log_atten'
epochs = args.epochs  # 100
pretrain_dir = args.pretrain_dir  # config.places_alex
ckpt_dir = args.ckpt_dir  # '/mnt/old_hexin/log/2019-04-26_20:42:46/checkpoints/best/epochs:92.ckpt'
baseline_dir = args.baseline_dir
log_folder = args.log_folder
lr = args.lr  # 3e-4
l2 = args.l2  # 0.1
atten_type = args.atten_type
optim_type = args.optim_type
optim_load = str2bool(args.optim_load)

# no_atten = str2bool(args.no_atten)

# configure the gpu
device = torch.device(args.device if args.device else 'cuda:1')

# setup dataset
phases = ['train', 'val', 'test']
trans = tr_hha, te_hha, te_hha
sun_data_args = [
    (
        (config.sunrgbd_dir, config.sunrgbd_label_dict_dir), dict(phase=phase, hha_mode=True, transform=tran)
    )
    for phase, tran in zip(phases, trans)
]
nyudv2_data_args = [
    (
        (config.nyud2_dir,), dict(phase=phase, hha_mode=True, transform=tran)
    )
    for phase, tran in zip(phases, trans)
]

# dataset = SunRgbdDataset
dataset = NYUD2Dataset
# data_args = sun_data_args
data_args = nyudv2_data_args

train_data, val_data, test_data = [dataset(*d_arg[0], **d_arg[1]) for d_arg in data_args]
train_loader = DataLoader(train_data, batch_size=125, shuffle=True, drop_last=False)  # , num_workers=10)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, drop_last=False)  # , num_workers=5)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False)  # , num_workers=5)
classes = train_data.classes

cls_weight = train_data.cls_weight

# define: the model
if model is None:
    # Experiment 1: full_freeze, with Atten (dot Attention)
    # set atten_type = 'raw' or 'multi' (although effective)
    #
    # model = AttenDBAlex(
    #     train_data.cls_count, pretrain_dir=pretrain_dir, baseline_dir=baseline_dir,
    #     atten_type=atten_type, freeze_front=True, freeze_z=False, zero_z=True
    # )

    # Experiment 2: full_freeze, with Atten (dot Attention) with Residual connection
    #
    # model = ResidualAttenDBAlex(
    #     train_data.cls_count,
    #
    #     pretrain_dir=pretrain_dir,
    #     baseline_dir=baseline_dir,
    #     freeze_front=True, freeze_z=False, zero_z=True
    # )

    # Experiment 3: full_freeze, with Atten (dot Attention) and ReLU
    # set atten_type = 'raw' or 'multi' (although effective)
    # set raw_atten_no_relu_dir = args.raw_atten_no_relu_dir
    #
    # model = AttenDBAlexRelu(
    #     train_data.cls_count,
    #
    #     pretrain_dir=pretrain_dir,
    #     baseline_dir=baseline_dir,
    #     AttenDBAlex_dir=raw_atten_no_relu_dir,
    #
    #     atten_type=atten_type, freeze_front=True, freeze_z=False, zero_z=True
    # )

    # Experiment 4: freeze only Convs, with Atten (dot Attention)
    # set atten_type = 'raw' or 'multi' (although effective)
    #
    # model = AttenDBAlex(
    #     train_data.cls_count, pretrain_dir=pretrain_dir, baseline_dir=baseline_dir,
    #     atten_type=atten_type, freeze_features=False, freeze_z=False, zero_z=True, add_atten_after_relu=1
    # )

    # Experiment 5: freeze only Convs, with Atten_Instructor
    # param: Network in [ InstructorAlex, OnlyInstructorAlex ]
    # set atten_top = 0.4 (proportional) or None (all) or 100 (fix length)
    # set s_channel = 512 (small) or 4096 (once get 64% accuracy, but too huge)
    #
    # model = Network(train_data.cls_count, atten_topk=0.4, pretrain_dir=pretrain_dir, baseline_dir=baseline_dir,
    #                 freeze_front=True, s_channel=512)

    # Experiment 0: baseline
    # param: freeze_front: freeze all the layers except the last one
    # param: freeze_features: freeze the Conv layers

    # freeze_front baseline, origin
    # command: python work.py --lr 3e-4 --l2 0.1 --log_root /home/hong/small_log_atten/nyudv2/freeze_baseline/debug --epochs 100000 --pretrain_dir /mnt/pub_workspace_2T/hong_data/AttentiveScenery/models/ckpt/alexnet_places365.pth.tar --save_inteval_epoch 1 --device cuda:2

    model = DoubleBranchAlex(train_data.cls_count, pretrain_dir=pretrain_dir, freeze_front=True)

model.to(device)

# define: loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=torch.tensor(cls_weight)).to(device)
if optim_type == 'adam':
    print('adam mode')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    lr_scheduler = None
else:  # optim_type == 'sgd':
    if (not lr) and (not l2):
        lr = 0.0001
        l2 = 0.0005
    else:
        if not lr:
            lr = 0.0001
        if not l2:
            l2 = 0.0005
    print('sgd mode: hyperparams are lr:{0}, l2:{1}'.format(lr, l2))
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.1)

# resume from last time if necessary:
epoch_offset = 0
step_offset = 0
if ckpt_dir:
    ckpt = torch.load(ckpt_dir, map_location=device)
    epoch_offset = ckpt['epoch_offset']
    step_offset = ckpt['step_offset']
    model.load_state_dict(ckpt['state_dict'])
if optim_load:
    optimizer.load_state_dict(ckpt['optim'])
print('resume from epoch: {}, step: {}'.format(epoch_offset - 1, step_offset - 1))

# just before work: setup logger
log_folder = log_folder if log_folder else str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
writer = SummaryWriter(log_root + '/{}'.format(log_folder))  # TensorboardX: https://zhuanlan.zhihu.com/p/35675109

# work, the main loop!
worker = Worker(model,
                optimizer,
                lr_scheduler,
                criterion,
                epochs,
                epoch_offset,
                step_offset,
                train_loader,
                val_loader,
                test_loader,
                device,
                writer,
                log_root,
                log_folder
                )

print(worker.work())
