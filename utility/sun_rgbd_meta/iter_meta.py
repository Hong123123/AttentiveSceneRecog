import config
import json
import os
import shutil
import scipy.io
from collections import OrderedDict
import numpy as np

dataset_name = '/SUNRGBD_HHA'
rootDir = config.sunrgbd_root + dataset_name
rootwildcard = '/n/fs/sun3d/data/SUNRGBD'


mat_dir = config.sunrgbd_root + '/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat'
mat = scipy.io.loadmat(mat_dir, squeeze_me=True, struct_as_record=False)

# hyperparameters:
target_list = mat['SUNRGBDMeta']


o_dict = OrderedDict()
cls_index = -1
for idx, struct in enumerate(target_list):
    full_rgb = struct.rgbpath.replace(rootwildcard, rootDir)
    full_depth = struct.depthpath.replace(rootwildcard, rootDir)

    rgb_base1 = full_rgb[:full_rgb.rfind('/')]
    base_dir = rgb_base1[:rgb_base1.rfind('/')]

    fulltarget = os.path.join(base_dir, 'scene.txt')
    with open(fulltarget, 'r') as file:
        cls_name = file.read()

    if o_dict.keys() and (cls_name in o_dict.keys()):  # known cls_name
        o_dict[cls_name]['N'] += 1
    else:  # unknown cls_name
        cls_index += 1
        o_dict[cls_name] = {'N': 1, 'label': cls_index}

    print('processing: {}'.format(idx+1))

count = idx + 1

nums = np.array([v['N'] for v in o_dict.values()])
top19 = nums.argsort()[-19:]

top_dict = OrderedDict({list(o_dict.keys())[idx]: list(o_dict.values())[idx] for idx in top19})

if False:
    save_name = 'hehe'
    save_json_name = '/mnt/old_hexin/AttentiveScenery/utility/sun_rgbd_meta/{}_{}.json'.format(save_name, count)
    with open(save_json_name, '+w') as jf:
        json.dump(o_dict, jf)
