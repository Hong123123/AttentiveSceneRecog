import config
import os
import torch
import argparse


def save_helper(file, basename, filename, maxnum=-1):
    os.makedirs(basename) if not os.path.exists(basename) else ''
    torch.save(file, basename + filename)
    if maxnum != -1:  # define how much files would be kept in the folder
        exfiles = {int(exfile.replace(':', '.').split('.')[-2]): exfile
                   for exfile in os.listdir(basename) if os.path.isfile(os.path.join(basename, exfile))}
        if len(exfiles) >= maxnum:
            remove_keys = sorted(exfiles.keys())[0:-maxnum]
            for rm_key in remove_keys:
                rm_name = os.path.join(basename, exfiles[rm_key])
                os.remove(rm_name)


def str2bool(v):
    return v.lower() in ('true')


def get_parameters():

    parser = argparse.ArgumentParser()

    # hyper parameter
    parser.add_argument('--log_root', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pretrain_dir', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)

    parser.add_argument('--no_atten', type=str, default='False')

    # system params
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--device', type=str, default='cuda:1')

    # params have default value
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-1)

    parser.add_argument('--eval_inteval_epoch', type=int, default=1)
    parser.add_argument('--log_folder', type=str, default=None)
    parser.add_argument('--save_inteval_epoch', type=int, default=None)

    # parse
    args = parser.parse_args()

    # integrity check
    if args.save_inteval_epoch:
        args.save_inteval_epoch = args.eval_inteval_epoch
    if args.pretrain_dir:
        if args.ckpt_dir:
            raise ValueError('--pretrain_dir and --ckpt_dir can\'t be present at the same time.')

    return args


if __name__ == '__main__':
    save_helper('', '/mnt/old_hexin/log/2019-04-26_20:42:46/checkpoints/', '', 1)
