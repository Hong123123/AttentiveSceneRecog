from dataset.sunrgbd_dataset import SunRgbdDataset
from dataset.transforms import train_transform_hha as tr_hha
from models.duplicated_Atten_alex import AttenAlex
import config
from torch.utils.data import DataLoader
from torch import nn, optim
from datetime import datetime
from tensorboardX import SummaryWriter
from utility.train_utils import str2bool

# from Worker import Worker
from utility.train_utils import save_helper
import torch


class Worker():
    def __init__(self, model, optimizer, criterion,  # model params
                 epochs, epoch_offset, step_offset,
                 train_loader, val_loader, test_loader,
                 device, writer, log_root, log_folder, best_prec1=0,
                 save=True,
                 save_best=True,
                 debug_mode=False,
                 debug_batch=5):
        self.model = model
        self.epochs = epochs
        self.epoch_offset = epoch_offset
        self.step_offset = step_offset
        self.abs_epoch = -1
        self.abs_step = -1
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.writer = writer
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        # self.eval_inteval_epoch = eval_inteval_epoch
        # self.save_inteval_epoch = save_inteval_epoch
        self.best_prec1 = best_prec1
        self.log_folder = log_folder
        self.log_root = log_root

        self.save = save
        self.save_best = save_best
        self.debug_mode = debug_mode
        self.debug_batch = debug_batch

    def train(self, if_log=True):
        # Main loop config: https://discuss.pytorch.org/t/interpreting-loss-value/17665/4
        self.model.train(True)  # the train mode

        # iterate the dataset
        num_images = 0
        running_loss = 0
        running_acc = 0

        for step, batch in enumerate(iter(self.train_loader)):
            if self.debug_mode:
                if step+1 > self.debug_batch:
                    break
            self.abs_step = step + self.step_offset
            self.step_offset = self.abs_step + 1

            rgb, depth, label = batch['rgb'].to(self.device), batch['depth'].to(self.device), batch['label'].to(
                self.device)
            batch_size = rgb.size(0)
            print(batch_size)
            num_images += batch_size

            # calculate output and loss
            import numpy as np
            rgb = depth
            rgb = rgb.numpy()
            print('shape:{}'.format(rgb.shape))
            print('mean of c0:{},c1:{},c2:{}'.format(np.mean(rgb[0, 0, ...].flatten()), np.mean(rgb[0, 1, ...].flatten()),
                                                    np.mean(rgb[0, 2, ...].flatten())))
            print('std of c0:{},c1:{},c2:{}'.format(np.std(rgb[0, 0, ...].flatten()), np.std(rgb[0, 1, ...].flatten()),
                                                    np.std(rgb[0,2,...].flatten())))
            [1]
            # o = self.model(rgb, depth)

            # # optimization
            # loss = self.criterion(o, label)  # + self.model.parameters().values()
            # accu = torch.sum(
            #     torch.argmax(o, dim=1) == label
            # )
            # running_loss += (loss * batch_size).item()
            # running_acc += accu.item()
            #
            # # back-propagation
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()

        #     if if_log:
        #         print('running at step {}'.format(step))
        #         print('logging at step {}'.format(self.abs_step))
        #         self.writer.add_scalar('Train/Running_Loss(steps)', loss.item(), self.abs_step)
        #         self.writer.add_scalar('Train/Running_Accu(steps)', float(accu.item()) / batch_size, self.abs_step)
        #         self.writer.add_scalar('Train/Sanity of wz',
        #                                float(torch.sum(torch.abs(self.model.attens[0][0].Wz[0].weight.flatten())).item()),
        #                                self.abs_step)
        #
        # return running_loss, running_acc, num_images
        return '', '', ''

    def validate(self, loader, mode='Validate', if_log=True):
        # validate:
        print(mode + ' at epoch{}'.format(self.abs_epoch))
        self.model.eval()  # switch to eval mode
        running_val_acc = 0
        val_num = 0
        for step, batch in enumerate(iter(loader)):
            if self.debug_mode:
                if step+1 > self.debug_batch:
                    break
            # global test_N
            # if step>5:  # testing code
            #     break
            with torch.no_grad():
                rgb, depth, label = batch['rgb'].to(self.device), batch['depth'].to(self.device), batch['label'].to(
                    self.device)

                o = self.model(rgb, depth)

                val_num += rgb.size(0)
                running_val_acc += torch.sum(
                    torch.argmax(o, dim=1) == label
                ).item()
        val_acc = running_val_acc / val_num

        is_best = val_acc >= self.best_prec1
        self.best_prec1 = val_acc if is_best else self.best_prec1

        if if_log:
            self.writer.add_scalar(mode + '/Accu(epochs)', val_acc, self.abs_epoch)
            self.writer.add_scalar(mode + '/Best_accu(epochs)', self.best_prec1, self.abs_epoch)
        return val_acc, is_best

    def save_switch(self, val_acc, is_best):
        if not ((not self.debug_mode) or val_acc > 0.8):
            return

        state_dict = self.model.state_dict()
        arch = self.model._get_name()

        # ckpt ref: https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/main.py
        ckpt = {
            'epoch_offset': self.abs_epoch + 1,
            'step_offset': self.abs_step + 1,
            'arch': arch,
            'state_dict': state_dict,
            'best_prec1': self.best_prec1,
            'optim': self.optimizer.state_dict(),
        }

        if self.debug_mode:
            ckpt['best_prec1'] = 0  # don't save this stats

        if is_best and self.save_best:
            print('saving best model at epoch: {}'.format(self.abs_epoch))
            best_dir = self.log_root + '/{}/checkpoints/best'.format(self.log_folder)
            save_name = '/val_acc_of_{}_at_epoch:{}.ckpt'.format(val_acc, self.abs_epoch)
            save_helper(ckpt, best_dir, save_name, maxnum=5)

        if not self.save:
            return

        # if is_save:
        print('saving at epoch: {}'.format(self.abs_epoch))
        save_dir = self.log_root + '/{}/checkpoints'.format(self.log_folder)
        save_name = '/val_acc_of_{}_at_epoch:{}.ckpt'.format(val_acc, self.abs_epoch)
        save_helper(ckpt, save_dir, save_name, maxnum=10)

    def work(self):
        for epoch in range(self.epochs):
            self.abs_epoch = epoch + self.epoch_offset
            # total_num_step=0

            running_loss, running_acc, num_images = self.train()

            # epoch_loss = running_loss / num_images
            # epoch_acc = float(running_acc) / num_images

            # self.writer.add_scalar('Train/Loss(epochs)', epoch_loss, self.abs_epoch)
            # self.writer.add_scalar('Train/Accu(epochs)', epoch_acc, self.abs_epoch)

            # val_acc, is_best = self.validate(self.val_loader)
            # print('val_acc: {}, is_best: {}'.format(val_acc, is_best))

            # self.save_switch(val_acc, is_best)

        # test
        # test_acc, _ = self.validate(self.test_loader, mode='Test')
        # return test_acc
        return ' '


def main(args):
    # hyperparameters
    log_root = args.log_root  # config.log_root  # '/mnt/old_hexin/log_atten'
    epochs = args.epochs  # 100
    pretrain_dir = args.pretrain_dir  # config.places_alex
    ckpt_dir = args.ckpt_dir  # '/mnt/old_hexin/log/2019-04-26_20:42:46/checkpoints/best/epochs:92.ckpt'
    log_folder = args.log_folder
    lr = args.lr  # 3e-4
    l2 = args.l2  # 0.1

    no_atten = str2bool(args.no_atten)

    # configure the gpu
    device = torch.device(args.device if args.device else 'cuda:1')

    # setup dataset
    train_data = SunRgbdDataset(
        config.sunrgbd_root, config.sunrgbd_label_dict_dir,
        phase='train', hha_mode=True, transform=tr_hha,
    )
    val_data = SunRgbdDataset(
        config.sunrgbd_root, config.sunrgbd_label_dict_dir,
        phase='val', hha_mode=True, transform=tr_hha
    )
    test_data = SunRgbdDataset(
        config.sunrgbd_root, config.sunrgbd_label_dict_dir,
        phase='test', hha_mode=True, transform=tr_hha
    )
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=5, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=5, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=5, drop_last=False)
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
                    # )
                    save=False,
                    save_best=False,
                    debug_mode=True,
                    debug_batch=2,
                    )

    print(worker.work())


if __name__ == '__main__':
    from utility.train_utils import get_parameters
    args = get_parameters()
    main(args)
