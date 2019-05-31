from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import scipy.io
import json
import glob
import config


class SunRgbdDataset(Dataset):
    def __init__(self, data_dir, label_dict_dir, transform=None, phase='train', hha_mode=False):
        """
        :param data_dir: directory with the dataset
        :param transform: Transformation function
        :param label_dict_dir: dir to dictionary of labels which looks like:
        {'label_string': {'label':label_idx, 'N': label_occurence, 'weight':label_weight}}
        dataset for SUN RGB-D {'image': PIL, 'depth': PIL, 'label': np.1darray}
        """
        self.dataset_dir = data_dir
        self.root_dir = data_dir[:data_dir.rfind('/')]
        assert phase in ['train', 'val', 'test', 'debug']
        self.phase = phase
        self.hha_mode = hha_mode

        # pre-calculated stats:
        self.total_len = 9504
        self.alltest_len = 4659
        # self.alltrain_len = 4845
        self.train_len = 2393
        self.val_len = 2452

        self.transform = transform

        # split={'alltest': DIRs, 'alltrain': DIRs, 'trainvalsplit': {'train': DIRs, 'val': DIRs}}
        # the DIRs in split has redundancy of len 16 before '/SUNRGBD' and 24 for that after '/SUNRGBD'
        # self.split_dir = os.path.join(data_dir, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
        self.split_redundancy = 24
        # self.dataset_name = '/SUNRGBD_HHA'
        self.splits_dir = config.root + '/utility/sun_rgbd_meta/make_9504/total_split.json'
        # self.splits = scipy.io.loadmat(self.splits_dir, squeeze_me=True, struct_as_record=False)
        # self.all_split = [*self.splits['alltrain'], *self.splits['alltest']]
        with open(self.splits_dir, 'r') as file:
            self.splits = json.load(file)

        self.cls_dict_dir = label_dict_dir
        with open(self.cls_dict_dir, 'r') as j:
            self.cls_dict = json.load(j)
        self.cls_count = len(self.cls_dict.keys())
        self.classes = list(self.cls_dict.keys())
        self.cls_weight = [sample_dict['weight'] for scene, sample_dict in self.cls_dict.items()]

    def __len__(self):
        if self.phase == 'train':  # all train: no val
            return self.train_len
        elif self.phase == 'val':
            return self.val_len
        elif self.phase == 'test':
            return self.alltest_len
        elif self.phase == 'debug':
            return 1

    def __getitem__(self, idx):
        depth_mode = '/hha' if self.hha_mode else '/depth_bfx'

        if self.phase == 'train':
            # deal with filename with wildcard
            rgb_dir = glob.glob(self.dataset_dir +
                                self.splits['trainvalsplit']['train'][idx][self.split_redundancy:] + '/image/*')[0]
            # the inpainted depth:
            depth_dir = glob.glob(self.dataset_dir +
                                  self.splits['trainvalsplit']['train'][idx][self.split_redundancy:] + depth_mode + '/*')[0]
            cls_dir = self.dataset_dir + \
                      self.splits['trainvalsplit']['train'][idx][self.split_redundancy:] + '/scene.txt'
        elif self.phase == 'val':
            # deal with filename with wildcard
            rgb_dir = glob.glob(self.dataset_dir +
                                self.splits['trainvalsplit']['val'][idx][self.split_redundancy:] + '/image/*')[0]
            # the inpainted depth:
            depth_dir = glob.glob(self.dataset_dir +
                                  self.splits['trainvalsplit']['val'][idx][self.split_redundancy:] + depth_mode + '/*')[0]
            cls_dir = self.dataset_dir + \
                      self.splits['trainvalsplit']['val'][idx][self.split_redundancy:] + '/scene.txt'
        elif self.phase == 'test':
            rgb_dir = glob.glob(self.dataset_dir +
                                self.splits['alltest'][idx][self.split_redundancy:] + '/image/*')[0]
            depth_dir = glob.glob(self.dataset_dir +
                                  self.splits['alltest'][idx][self.split_redundancy:] + depth_mode + '/*')[0]
            cls_dir = self.dataset_dir + \
                      self.splits['alltest'][idx][self.split_redundancy:] + '/scene.txt'
        elif self.phase == 'debug':
            # deal with filename with wildcard
            rgb_dir = glob.glob(self.dataset_dir +
                                self.splits['trainvalsplit']['train'][0][self.split_redundancy:] + '/image/*')[0]
            # the inpainted depth:
            depth_dir = glob.glob(self.dataset_dir +
                                  self.splits['trainvalsplit']['train'][0][self.split_redundancy:] + depth_mode + '/*')[0]
            cls_dir = self.dataset_dir + \
                      self.splits['trainvalsplit']['train'][0][self.split_redundancy:] + '/scene.txt'
        rgb_image = Image.open(rgb_dir)  # np array of shape(H,W,C=3)
        depth = Image.open(depth_dir)  # np array of shape(H,W) if not hha_mode else shape(H,W,C=3)
        # sample = {'rgb': rgb_image, 'depth': depth_image}
        with open(cls_dir,'r') as l:
            cls_name = l.read()

        cls_idx = np.int(self.cls_dict[cls_name]['label'])
        label = cls_idx

        if self.transform:
            t1,t2,t3 = self.transform
            if t1:
                rgb_image = t1(rgb_image)
            if t2:
                depth = t2(depth)
            if t3:
                label = t3(label)

        sample = {'rgb': rgb_image, 'depth': depth, 'label': label}
        return sample


if __name__ == '__main__':
    import config
    from dataset.transforms import train_transform
    dataset = SunRgbdDataset(config.sunrgbd_dir, config.sunrgbd_label_dict_dir, transform=train_transform)
    one_sample = dataset[0]
    pass
