from dataset.sunrgbd_dataset import SunRgbdDataset
from dataset.transforms import train_transform_hha as tr_hha
from models.duplicated_Atten_alex import AttenAlex
import config
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from datetime import datetime
from tensorboardX import SummaryWriter

from Worker import Worker


def main(args):
    # hyperparameters
    log_root = args.log_root  # config.log_root  # '/mnt/old_hexin/log_atten'
    epochs = args.epochs  # 100
    pretrain_dir = args.pretrain_dir  # config.places_alex
    ckpt_dir = args.ckpt_dir  # '/mnt/old_hexin/log/2019-04-26_20:42:46/checkpoints/best/epochs:92.ckpt'
    log_folder = args.log_folder
    lr = args.lr  # 3e-4
    l2 = args.l2  # 0.1

    split = args.split  # train split or all split

    no_atten = args.no_atten

    # configure the gpu
    device = torch.device(args.device if args.device else 'cuda:1')

    # setup dataset
    train_data = SunRgbdDataset(
        config.sunrgbd_root, config.sunrgbd_label_dict_dir,
        phase='train', hha_mode=True, transform=tr_hha,
    )
    val_data = SunRgbdDataset(
        config.sunrgbd_root, config.sunrgbd_label_dict_dir,
        phase='train', hha_mode=True, transform=tr_hha
    )
    test_data = SunRgbdDataset(
        config.sunrgbd_root, config.sunrgbd_label_dict_dir,
        phase='debug', hha_mode=True, transform=tr_hha
    )
    train_loader = DataLoader(train_data, batch_size=20, shuffle=False, num_workers=5, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=20, shuffle=False, num_workers=5, drop_last=False)
    test_loader = DataLoader(val_data, batch_size=20, shuffle=False, num_workers=5, drop_last=False)
    cls_dict = train_data.cls_dict
    classes = list(cls_dict.keys())
    cls_weight = [sample_dict['frequency'] for scene,sample_dict in train_data.cls_dict.items()]

    # define the model, loss function and optimizer
    model = AttenAlex(45).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cls_weight)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)

    # modify the model
    if no_atten:
        for atten_param in model.attens.parameters():
            atten_param.requires_grad = False

    # read pretrained weight if necessary
    if pretrain_dir:
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

    # ready to work
    log_folder = log_folder if log_folder else str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    writer = SummaryWriter(log_root + '/{}'.format(log_folder))  # TensorboardX: https://zhuanlan.zhihu.com/p/35675109

    # work !
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

                    save_regular=True,
                    save_best=True,
                    debug_mode=True,
                    )

    worker.work()


if __name__ == '__main__':
    from utility.train_utils import get_parameters
    args = get_parameters()
    main(args)
