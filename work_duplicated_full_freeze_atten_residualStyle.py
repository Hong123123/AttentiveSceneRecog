# from dataset.sunrgbd_dataset import SunRgbdDataset
from dataset.nyud2_dataset import NYUD2Dataset
from dataset.transforms import train_transform_hha as tr_hha, test_transform_hha as te_hha
# from models.Atten_alex import AttenAlex
from models.dynamic_atten_alex import DoubleBranchAlex, AttenDBAlex, AttenDBAlexRelu, ResidualAttenDBAlex
import config
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from datetime import datetime
from tensorboardX import SummaryWriter
from utility.train_utils import str2bool

from Worker import Worker


def main(args, model=None):
    # hyperparameters
    log_root = args.log_root  # config.log_root  # '/mnt/old_hexin/log_atten'
    epochs = args.epochs  # 100

    pretrain_dir = args.pretrain_dir  # config.places_alex
    ckpt_dir = args.ckpt_dir  # '/mnt/old_hexin/log/2019-04-26_20:42:46/checkpoints/best/epochs:92.ckpt'
    baseline_dir = args.baseline_dir
    raw_atten_no_relu_dir = args.raw_atten_no_relu_dir

    log_folder = args.log_folder
    lr = args.lr  # 3e-4
    l2 = args.l2  # 0.1
    atten_type = args.atten_type



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
    train_loader = DataLoader(train_data, batch_size=125, shuffle=True, drop_last=False)  #, num_workers=1)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, drop_last=False)  #, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False)  #, num_workers=1)
    classes = train_data.classes
    cls_weight = train_data.cls_weight

    # define the model, loss function and optimizer
    # >>>> freeze grad
    if model is None:
        model = ResidualAttenDBAlex(
            train_data.cls_count,

            pretrain_dir=pretrain_dir,
            baseline_dir=baseline_dir,
            freeze_front=True, freeze_z=False, zero_z=True
        )
        # model = DAlex(train_data.cls_count, pretrain_dir=pretrain_dir, freeze_front=True)

    model.to(device)

    # for cd in list(model.children())[:-1]:
    #     for param in cd.parameters():
    #         param.requires_grad=False
    #     print('-'*10, 'freezing', '-'*10, cd)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cls_weight)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    # resume from last time if necessary:
    epoch_offset = 0
    step_offset = 0
    if ckpt_dir or baseline_dir:
        dir = ckpt_dir if ckpt_dir else baseline_dir
        ckpt = torch.load(dir, map_location=device)
        epoch_offset = ckpt['epoch_offset']
        step_offset = ckpt['step_offset']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optim'])
        print('resume from epoch: {}, step: {}'.format(epoch_offset - 1, step_offset - 1))

    # just before work: setup logger
    log_folder = log_folder if log_folder else str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    writer = SummaryWriter(log_root + '/{}'.format(log_folder))  # TensorboardX: https://zhuanlan.zhihu.com/p/35675109

    # work, the main loop!
    worker = Worker(model,
                    optimizer,
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
                    log_folder,
                    # save_first=False,
                    # save_regular=False,
                    # save_best=False
                    )

    print(worker.work())


if __name__ == '__main__':
    from utility.train_utils import get_parameters

    args = get_parameters()

    # model = AttenDAlex(10, atten_type='raw', freeze_front=True, freeze_z=True, zero_z=True)
    main(args)
