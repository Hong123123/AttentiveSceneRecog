# from dataset.sunrgbd_dataset import SunRgbdDataset
from dataset.nyud2_dataset import NYUD2Dataset
from dataset.transforms import train_transform_hha as tr_hha
# from models.Atten_alex import AttenAlex
from models.origin_alex import DAlex
import config
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from datetime import datetime
from tensorboardX import SummaryWriter
from utility.train_utils import str2bool

from Worker import Worker


def load_pretrain_both_branch(model, pretrain_dir, device):
    pre_ckpt = torch.load(pretrain_dir, map_location=device)

    # load weight manually
    # rather than model.load_state_dict(state_dict)
    conv_idxmap = model.where_convs
    key1 = None
    key2 = None
    for k, (key, value) in enumerate(pre_ckpt['state_dict'].items()):
        # key = str.replace(key, 'module.', '')  # preprocssing]
        if key.find('features') > -1:
            # key = str.replace(key, 'features', 'r_features')  # transfering Places2Net features
            if k < 10:
                if k % 2 == 0:
                    key1 = 'r_features.{}.weight'.format(conv_idxmap[k // 2])
                    key2 = 'd_features.{}.weight'.format(conv_idxmap[k // 2])
                else:
                    key1 = 'r_features.{}.bias'.format(conv_idxmap[k // 2])
                    key2 = 'd_features.{}.bias'.format(conv_idxmap[k // 2])
        else:
            target = 'classifier'  # transfering Places2Net classifiers
            sp = key.split('.')
            assert len(sp) == 3
            if sp[0] == target:
                key1 = 'r_other.1.{}.{}'.format(int(sp[1]) + 1, sp[2])
                key2 = 'd_other.1.{}.{}'.format(int(sp[1]) + 1, sp[2])
        if key1 in model.state_dict().keys():
            print('copied: {}'.format(key))
            model.state_dict()[key1].copy_(value)
        if key2 in model.state_dict().keys():
            print('copied: {}'.format(key))
            model.state_dict()[key2].copy_(value)


def load_pretrain_rgb_branch(model, pretrain_dir, device):
    pre_ckpt = torch.load(pretrain_dir, map_location=device)

    # load weight manually
    # rather than model.load_state_dict(state_dict)
    conv_idxmap = model.where_convs
    for k, (key, value) in enumerate(pre_ckpt['state_dict'].items()):
        # key = str.replace(key, 'module.', '')  # preprocssing]
        if key.find('features') > -1:
            # key = str.replace(key, 'features', 'r_features')  # transfering Places2Net features
            if k < 10:
                if k % 2 == 0:
                    key = 'r_features.{}.weight'.format(conv_idxmap[k // 2])
                else:
                    key = 'r_features.{}.bias'.format(conv_idxmap[k // 2])
        else:
            target = 'classifier'  # transfering Places2Net classifiers
            sp = key.split('.')
            assert len(sp) == 3
            if sp[0] == target:
                key = 'r_other.1.{}.{}'.format(int(sp[1]) + 1, sp[2])
        if key in model.state_dict().keys():
            print('copied: {}'.format(key))
            model.state_dict()[key].copy_(value)


def load_pretrain_alex(model, pretrain_dir, device):
    pass


def main(args):
    # hyperparameters
    log_root = args.log_root  # config.log_root  # '/mnt/old_hexin/log_atten'
    epochs = args.epochs  # 100
    pretrain_dir = args.pretrain_dir  # config.places_alex
    ckpt_dir = args.ckpt_dir  # '/mnt/old_hexin/log/2019-04-26_20:42:46/checkpoints/best/epochs:92.ckpt'
    log_folder = args.log_folder
    lr = args.lr  # 3e-4
    l2 = args.l2  # 0.1

    # no_atten = str2bool(args.no_atten)

    # configure the gpu
    device = torch.device(args.device if args.device else 'cuda:1')

    # setup dataset
    phases = ['train', 'val', 'test']
    sun_data_args = [
        (
            (config.sunrgbd_dir, config.sunrgbd_label_dict_dir), dict(phase=phase, hha_mode=True, transform=tr_hha)
        )
        for phase in phases
    ]
    nyudv2_data_args = [
        (
            (config.nyud2_dir,), dict(phase=phase, hha_mode=True, transform=tr_hha)
        )
        for phase in phases
    ]

    # dataset = SunRgbdDataset
    dataset = NYUD2Dataset
    # data_args = sun_data_args
    data_args = nyudv2_data_args

    train_data, val_data, test_data = [dataset(*d_arg[0], **d_arg[1]) for d_arg in data_args]
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True, num_workers=5, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=20, shuffle=False, num_workers=5, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=20, shuffle=False, num_workers=5, drop_last=False)
    classes = train_data.classes
    cls_weight = train_data.cls_weight

    # define the model, loss function and optimizer
    model = DAlex(train_data.cls_count, pretrain_dir=pretrain_dir).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cls_weight)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)

    # resume from last time if necessary:
    epoch_offset = 0
    step_offset = 0
    if ckpt_dir:
        ckpt = torch.load(ckpt_dir, map_location=device)
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
                    )

    print(worker.work())


if __name__ == '__main__':
    from utility.train_utils import get_parameters
    args = get_parameters()
    main(args)
