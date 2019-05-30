import config
import os
import scipy.io
from collections import OrderedDict
import numpy as np

dataset_name = '/SUNRGBD_HHA'
rootDir = config.sunrgbd_root + dataset_name
rootwildcard = '/n/fs/sun3d/data/SUNRGBD'


mat_dir = config.sunrgbd_root + '/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat'
mat = scipy.io.loadmat(mat_dir, squeeze_me=True, struct_as_record=False)


# alltrain, alltest, trainvalsplit.train, trainvalsplit.val

raw_list = mat['alltrain']
# raw_list = mat['alltest']
# raw_list = mat['trainvalsplit'].train
# raw_list = mat['trainvalsplit'].val

o_dict = OrderedDict()
cls_index = -1
for idx, raw in enumerate(raw_list):
    base_dir = raw.replace(rootwildcard, rootDir)

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
sum = 0
for v in top_dict.values():
    sum += v['N']
print('total of {} samples'.format(sum))

if False:
    save_name = 'hehe'
    save_json_name = '/mnt/old_hexin/AttentiveScenery/utility/sun_rgbd_meta/{}_{}.json'.format(save_name, count)
    with open(save_json_name, '+w') as jf:
        json.dump(o_dict, jf)
