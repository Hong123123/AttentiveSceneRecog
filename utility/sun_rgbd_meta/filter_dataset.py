import config
import json
import os
import shutil
import scipy.io
from collections import OrderedDict
import numpy as np

dataset_name = '/SUNRGBD_HHA'
rootDir = config.sunrgbd_root + dataset_name
destination = '/mnt/pub_workspace_2T/hong_data/SUNRGBD_root/SUNRGBD_HHA_filtered'
target = 'scene.txt'

sfile = '/mnt/old_hexin/AttentiveScenery/utility/sunrgbd_label_fre_dict_10335_1083_1.json'
with open(sfile, 'r') as jload:
    o_dic = json.load(jload)

oo_dict = OrderedDict(o_dic)
nums = np.array([v['N'] for v in oo_dict.values()])
common19 = nums.argsort()[-19:].tolist()

filtered_dicts = [oo_dict[list(oo_dict.keys())[idx]] for idx in common19]
sum = 0
for d in filtered_dicts:
    sum+=d['N']

split_redundancy = 24
splits_dir = config.sunrgbd_root + '/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat'
splits = scipy.io.loadmat(splits_dir, squeeze_me=True, struct_as_record=False)

# hyperparameters:
what_split = 'trainvalsplit'  # 'alltrain', 'alltest', 'trainvalsplit'
target_list = splits[what_split].val
save_name = 'trainvalsplit_val'

new_list = []
count = 0
for idx, item in enumerate(target_list):
    base_dir = rootDir + item[split_redundancy:]
    fulltarget = os.path.join(base_dir, target)
    with open(fulltarget, 'r') as file:
        label = file.read()

    if o_dic[label]['N'] >= 80:  # what we need
        new_list.append(item)
        foldername = base_dir.split('/')[-1]
        current_des = destination + base_dir[len(rootDir):]  # -len(foldername)]

        # os.makedirs(current_des) if not os.path.exists(current_des) else ''

        # full copy
        # shutil.copytree(base_dir, current_des)

        count += 1
        print('processing: {}'.format(count))

save_json_name = '/mnt/old_hexin/AttentiveScenery/utility/sun_rgbd_meta/{}_{}.json'.format(save_name, count)
with open(save_json_name, '+w') as jf:
    json.dump(new_list, jf)
