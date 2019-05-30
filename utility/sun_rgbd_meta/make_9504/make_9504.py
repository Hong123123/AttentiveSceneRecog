import config
import os
import scipy.io
from collections import OrderedDict
import numpy as np

dataset_name = '/SUNRGBD_HHA'
rootDir = config.sunrgbd_root + dataset_name
rootwildcard = '/n/fs/sun3d/data/SUNRGBD'
df2net_hha_list_name = '/mnt/old_hexin/AttentiveSceneReco/utility/sun_rgbd_meta/make_9504/sun_9504.txt'

mat_dir = config.sunrgbd_root + '/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat'
mat = scipy.io.loadmat(mat_dir, squeeze_me=True, struct_as_record=False)

save_name = 'trainvalsplit.val'
# alltrain, alltest, trainvalsplit.train, trainvalsplit.val

# raw_list = mat['alltrain']
# raw_list = mat['alltest']
# raw_list = mat['trainvalsplit'].train
raw_list = mat['trainvalsplit'].val

# load df^2 net hha list
# >>>
df2net_essence_list = []
with open(df2net_hha_list_name, 'r') as file:
    for line in file.readlines():
        # line = line.replace('drive/My Drive/sun_hha', '')
        # line = line[:line.rfind('/')]
        df2net_essence_list.append(line)

o_list = []
cls_index = -1
count = 0
for idx, raw in enumerate(raw_list):
    base_dir = raw.replace(rootwildcard, rootDir)
    essence = raw.replace(rootwildcard, '')
    # filter
    found = False
    for dfess in df2net_essence_list:
        if essence in str(dfess):
            found = True

    # fulltarget = os.path.join(base_dir, 'scene.txt')
    # with open(fulltarget, 'r') as file:
    #     cls_name = file.read()
    #
    # if o_dict.keys() and (cls_name in o_dict.keys()):  # known cls_name
    #     o_dict[cls_name]['N'] += 1
    # else:  # unknown cls_name
    #     cls_index += 1
    #     o_dict[cls_name] = {'N': 1, 'label': cls_index}

    if found:
        count += 1
        o_list.append(raw)
    else:
        print(essence)

    print('picked: {}, out of {}'.format(count, idx+1))
print('collected {}'.format(len(o_list)))

if True:
    import json
    save_fullname = '/mnt/old_hexin/AttentiveSceneReco/utility/sun_rgbd_meta/make_9504/{}_{}.split'.format(save_name, len(o_list))
    with open(save_fullname, 'w+') as file:
        json.dump(o_list, file)
