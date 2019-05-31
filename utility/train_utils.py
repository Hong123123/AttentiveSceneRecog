import config
import os
import torch
import argparse
import json


def save_helper(ckpt_obj, basename, filename, maxnum=-1,
                confusion_matrix=None, confusion_basename=None, confusion_filename=None):
    confused = None
    if confusion_matrix:
        assert confusion_basename
        assert confusion_filename
        confused = True

    os.makedirs(basename) if not os.path.exists(basename) else ''
    if confused:
        os.makedirs(confusion_basename) if not os.path.exists(confusion_basename) else ''

    torch.save(ckpt_obj, basename + filename)
    if confused:
        with open(confusion_basename + confusion_filename, 'w+') as c_file:
            json.dump(confusion_matrix, c_file)

    if maxnum != -1:  # define how much files would be kept in the folder
        exfiles = {int(exfile.replace(':', '.').split('.')[-2]): exfile  # epoch_num: file names
                   for exfile in os.listdir(basename) if os.path.isfile(os.path.join(basename, exfile))}
        if confused:
            exfiles_confusion = {int(exfile.replace(':', '.').split('.')[-2]): exfile  # epoch_num: file names
                       for exfile in os.listdir(confusion_basename) if os.path.isfile(os.path.join(confusion_basename, exfile))}
        if len(exfiles) >= maxnum:
            remove_keys = sorted(exfiles.keys())[0:-maxnum]
            for rm_key in remove_keys:
                rm_name = os.path.join(basename, exfiles[rm_key])
                os.remove(rm_name)

                if confused:
                    rm_confusion_name = os.path.join(confusion_basename, exfiles_confusion[rm_key])
                    os.remove(rm_confusion_name)


def str2bool(v):
    return v.lower() in 'true'


def get_parameters():

    parser = argparse.ArgumentParser()

    # hyper parameter
    parser.add_argument('--log_root', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)

    # one dir at the same time
    parser.add_argument('--pretrain_dir', type=str, default=None)
    parser.add_argument('--baseline_dir', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--raw_atten_no_relu_dir', type=str, default=None)

    parser.add_argument('--atten_type', type=str, default=None)

    # system params
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--device', type=str, default='cuda:1')

    # optimizer: adam or sgd
    parser.add_argument('--optim_type', type=str, default='adam')
    parser.add_argument('--optim_load', type=str, default='true')

    # params have default value
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--l2', type=float, default=None)

    parser.add_argument('--eval_inteval_epoch', type=int, default=1)
    parser.add_argument('--log_folder', type=str, default=None)
    parser.add_argument('--save_inteval_epoch', type=int, default=None)

    parser.add_argument('--save_num', type=int, default=None)
    parser.add_argument('--save_best_num', type=int, default=None)

    # parse
    args = parser.parse_args()

    # integrity check
    if args.save_inteval_epoch:
        args.save_inteval_epoch = args.eval_inteval_epoch

    if bool(args.pretrain_dir) + bool(args.baseline_dir) + bool(args.ckpt_dir) + bool(args.raw_atten_no_relu_dir) > 1:
        raise ValueError('multiple dirs can\'t be present at the same time.')

    if args.atten_type:
        if args.atten_type not in ['raw', 'multi']:
            raise ValueError('Unexpected --atten_type value')

    if args.optim_type:
        if args.optim_type not in ['sgd', 'adam']:
            raise ValueError('Unexpected --optim_type value')

    return args


if __name__ == '__main__':
    save_helper('', '/mnt/old_hexin/log/2019-04-26_20:42:46/checkpoints/', '', 1)
