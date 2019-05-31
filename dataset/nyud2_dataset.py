from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import scipy.io
import json
import glob
from utility.nyud2_meta.make_common_10 import sample_sceneTypes, sample_sceneIndices,\
    scene_names, scene_weight_10, scene_map


class NYUD2Dataset(Dataset):
    def __init__(self, data_dir,
                 transform=None, phase='train', hha_mode=True):
        """mapped
        :param data_dir: Parent directory with the dataset
        :param transform: Transformation function
        dataset for SUN RGB-D {'image': PIL, 'depth': PIL, 'label': np.1darray}
        """
        self.dataset_dir = data_dir
        self.splits_dir = '/mnt/pub_workspace_2T/hong_data/NYUv2_ROOT/nyud_splits.mat'

        assert phase in ['train', 'val', 'test', 'debug']
        self.phase = phase
        self.hha_mode = hha_mode
        self.cls_weight = scene_weight_10
        self.classes = scene_names

        self.transform = transform

        # keys: 'trainNdxs', 'testNdxs' ! (no val spilt provided)
        self.splits = scipy.io.loadmat(self.splits_dir, squeeze_me=True, struct_as_record=False)
        self.train_split = self.splits['trainNdxs']
        self.test_split = self.splits['testNdxs']
        self.val_split = self.splits['testNdxs']

        self.cls_count = 10  # common 10

    def __len__(self):
        if self.phase == 'train':
            return len(self.train_split)
        elif self.phase == 'val':
            return len(self.test_split)
        elif self.phase == 'test':
            return len(self.test_split)
        elif self.phase == 'debug':
            return 1

    def __getitem__(self, idx):
        if self.phase == 'train':
            split_mapped_idx = self.train_split[idx]
        elif self.phase == 'val':
            split_mapped_idx = self.val_split[idx]
        elif self.phase == 'test':
            split_mapped_idx = self.test_split[idx]
        else:
            split_mapped_idx = 1
        # deal with filename with wildcard
        foldernames = ['image', 'hha' if self.hha_mode else 'depth_bfx']
        rgb_dir, depth_dir = [glob.glob(self.dataset_dir + '/NYU{:04d}'.format(split_mapped_idx) + '/{}/*'.format(folder))[0]
                              for folder in foldernames]

        sun_cls_dir = self.dataset_dir + '/NYU{:04d}'.format(split_mapped_idx) + '/scene.txt'

        rgb_image, depth = Image.open(rgb_dir), Image.open(depth_dir)  # np array of shape(H,W,C=3) or shape(H,W)

        with open(sun_cls_dir, 'r') as lf:  # no need to read anymore
            sun_name = lf.read()
        nyu_name = sample_sceneTypes[split_mapped_idx - 1]

        # if sun_name != nyu_name:
        #     print('s: {}, n: {}'.format(sun_name, nyu_name))

        cls_idx = scene_map[sample_sceneIndices[split_mapped_idx - 1]]  # common 10
        # cls_name = sample_sceneTypes[self.train_split[idx] - 1]
        label = cls_idx

        if self.transform:
            t1,t2,t3 = self.transform
            if t1:
                rgb_image = t1(rgb_image)
            if t2:
                depth = t2(depth)
            if t3:
                label = t3(label)

        # print('getting item:\n {}'.format(rgb_dir))
        return {'rgb': rgb_image, 'depth': depth, 'label': label}


if __name__ == '__main__':
    import config
    from dataset.transforms import train_transform
    dataset = NYUD2Dataset(config.nyud2_dir, phase='val', transform=train_transform)
    one_sample = dataset[0]
    for i, sample in enumerate(dataset):
        pass
    pass
