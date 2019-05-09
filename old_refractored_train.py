from dataset.sunrgbd_dataset import SunRgbdDataset
from dataset.transforms import train_transform_hha as tr_hha
from models.Atten_alex import AttenAlex
from utility.train_utils import save_helper
import config
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from datetime import datetime
from tensorboardX import SummaryWriter


def main(args):
    # hyperparameters
    log_root = args.log_root  # config.log_root  # '/mnt/old_hexin/log_atten'
    epochs = args.epochs  # 100
    pretrain_dir = args.pretrain_dir  # ''  # config.places_alex
    ckpt_dir = args.ckpt_dir  # '/mnt/old_hexin/log/2019-04-26_20:42:46/checkpoints/best/epochs:92.ckpt'
    eval_inteval_epoch = args.eval_inteval_epoch
    save_inteval_epoch = args.save_inteval_epoch
    log_folder = args.log_folder
    lr = args.lr  # 3e-4

    split = args.split  # train split or all split

    no_atten = args.no_atten

    # configure the gpu
    device = torch.device(args.device if args.device else 'cuda:1')

    # setup dataset
    train_data = SunRgbdDataset(
        config.sunrgbd_root, config.sunrgbd_label_dict_dir,
        phase=split, hha_mode=True, transform=tr_hha,
    )
    test_data = SunRgbdDataset(
        config.sunrgbd_root, config.sunrgbd_label_dict_dir,
        phase='test', hha_mode=True, transform=tr_hha
    )
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True, num_workers=1, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=20, shuffle=False, num_workers=1, drop_last=False)
    cls_dict = train_data.cls_dict
    classes = list(cls_dict.keys())
    cls_weight = [sample_dict['frequency'] for scene,sample_dict in train_data.cls_dict.items()]

    # define the model, loss function and optimizer
    model = AttenAlex(45).to(device)
    cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor(cls_weight)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # modify the model
    if no_atten:
        for atten_param in model.attens.parameters():
            atten_param.requires_grad = False

    # read pretrained weight if necessary
    epoch_offset = 0
    step_offset = 0
    best_prec1 = 0

    if pretrain_dir:
        pre_ckpt = torch.load(pretrain_dir, map_location=device)

        # load weight manually
        # rather than model.load_state_dict(state_dict)
        for k, v in pre_ckpt['state_dict'].items():
            k = str.replace(k, 'module.', '')  # preprocssing

            target1 = 'features'
            if k.find(target1) > -1:
                k = str.replace(k, target1, 'r_features')  # transfering Places2Net features
            else:
                target2 = 'classifier'  # transfering Places2Net classifiers
                sp = k.split('.')
                assert len(sp) == 3
                if sp[0] == target2:
                    k = 'r_other.1.{}.{}'.format(int(sp[1]) + 1, sp[2])
            if k in model.state_dict().keys():
                model.state_dict()[k].copy_(v)  # need to use .copy_()

    # resume from last time if necessary:
    if ckpt_dir:
        # ckpt: {
        #     'epoch_offset': abs_epoch + 1,
        #     'step_offset': abs_step + 1,
        #     'arch': arch,
        #     'state_dict': state_dict,
        #     'best_prec1': best_prec1,
        #     'optim': optimizer.state_dict(),
        # }
        ckpt = torch.load(ckpt_dir, map_location=device)
        epoch_offset = ckpt['epoch_offset']
        step_offset = ckpt['step_offset']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optim'])
        print('resume from epoch: {}, step: {}'.format(epoch_offset - 1, step_offset - 1))

    # main loop
    log_folder = log_folder if log_folder else str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    writer = SummaryWriter(log_root + '/{}'.format(log_folder))  # TensorboardX: https://zhuanlan.zhihu.com/p/35675109

    step = 0  # global step
    for epoch in range(epochs):
        # Main loop config: https://discuss.pytorch.org/t/interpreting-loss-value/17665/4
        model.train(True)  # the train mode

        # iterate the dataset
        num_images = 0
        running_loss = 0
        running_acc = 0

        abs_epoch = epoch + epoch_offset
        abs_step = step + step_offset
        for batch in iter(train_loader):
            step += 1
            abs_step += 1

            rgb, depth, label = batch['rgb'].to(device), batch['depth'].to(device), batch['label'].to(device)
            batch_size = rgb.size(0)
            num_images += batch_size

            # calculate output and loss
            o = model(rgb, depth)

            # optimization
            loss = cross_entropy(o, label)
            accu = torch.sum(
                torch.argmax(o, dim=1)==label
            )
            running_loss += loss.item() * batch_size
            running_acc += accu.item()

            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # always log
            print('stepping at {}'.format(abs_step))
            writer.add_scalar('Train/Running_Loss(steps)', loss.item(), abs_step+1)
            writer.add_scalar('Train/Running_Accu(steps)', float(accu.item()) / batch_size, abs_step + 1)
            writer.add_scalar('Train/Sanity of wz', float(torch.sum(torch.abs(model.attens[0][0].Wz.weight.flatten())).item()), abs_step+1)

        # print(type(n), n)
        # print(type(acm), acm)
        epoch_loss = running_loss/num_images
        # print('ep_loss:', epoch_loss)
        epoch_acc = float(running_acc)/num_images
        # print('ep_acc:', epoch_acc)

        writer.add_scalar('Train/Loss(epochs)', epoch_loss, abs_epoch + 1)
        writer.add_scalar('Train/Accu(epochs)', epoch_acc, abs_epoch + 1)

        # evaluate if necessary:
        if epoch % eval_inteval_epoch == 0:
            print('evaluating for epoch{}'.format(abs_epoch))
            model.eval()  # switch to eval mode
            running_test_acc = 0
            test_N = 0
            for k,batch in enumerate(iter(test_loader)):
                # global test_N
                # if k>5:  # testing code
                #     break
                with torch.no_grad():
                    rgb, depth, label = batch['rgb'].to(device), batch['depth'].to(device), batch['label'].to(device)

                    o = model(rgb, depth)

                    test_N += rgb.size(0)
                    running_test_acc += torch.sum(
                        torch.argmax(o, dim=1) == label
                    ).item()
            test_acc = running_test_acc/test_N

            is_best = test_acc > best_prec1
            best_prec1 = test_acc if is_best else best_prec1

            writer.add_scalar('Test/Accu(epochs)', test_acc, abs_epoch+1)
            writer.add_scalar('Test/Best_accu(epochs)', best_prec1, abs_epoch + 1)

            is_save = epoch % save_inteval_epoch == 0

            if is_best or is_save:
                # save model if necessary
                # if epoch % save_inteval == 0:
                # dict_keys(['state_dict', 'epoch', 'arch', 'best_prec1'])
                state_dict = model.state_dict()
                arch = model._get_name()

                # ckpt ref: https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/main.py
                ckpt = {
                    'epoch_offset': abs_epoch + 1,
                    'step_offset': abs_step + 1,
                    'arch': arch,
                    'state_dict': state_dict,
                    'best_prec1': best_prec1,
                    'optim': optimizer.state_dict(),
                }
                if is_best:
                    print('saving best model at epoch: {}'.format(abs_epoch))
                    best_dir = log_root + '/{}/checkpoints/best'.format(log_folder)
                    save_name = '/epochs:{}.ckpt'.format(abs_epoch+1)
                    save_helper(ckpt, best_dir,save_name , maxnum=5)

                if is_save:
                    print('saving at epoch: {}'.format(abs_epoch))
                    save_dir = log_root+'/{}/checkpoints'.format(log_folder)
                    save_name = '/epochs:{}.ckpt'.format(abs_epoch+1)
                    save_helper(ckpt, save_dir, save_name, maxnum=10)


if __name__ == '__main__':
    from utility.train_utils import get_parameters
    args = get_parameters()
    main(args)
