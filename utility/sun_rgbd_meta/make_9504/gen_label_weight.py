import os
from collections import OrderedDict
import json
import config

# rootDir = config.sunrgbd_root
rootDir = '/mnt/pub_workspace_2T/hong_data/SUNRGBD_root/SUNRGBD'
target = 'seg.mat'
sceneName = 'scene.txt'
o_dict = OrderedDict()
raw_wildcard = '/n/fs/sun3d/data/SUNRGBD'

alltrain_name = '/mnt/old_hexin/AttentiveSceneReco/utility/sun_rgbd_meta/make_9504/alltrain_4845.split'
alltest_name = '/mnt/old_hexin/AttentiveSceneReco/utility/sun_rgbd_meta/make_9504/alltest_4659.split'

with open(alltrain_name, 'r') as file:
    tajson = json.load(file)
with open(alltest_name, 'r') as file:
    tejson = json.load(file)

split = [*tajson, *tejson]

index = 0
count = 0
for idx, raw in enumerate(split):
    count += 1
    base_dir = raw.replace(raw_wildcard, rootDir)
    with open(os.path.join(base_dir, sceneName), 'r') as file:
        scene=file.read()
    if scene not in o_dict.keys():
        o_dict[scene] = {'label': index, 'N': 1}
        index += 1
        # print('formulating dictionary {:.2%}'.format(count/num))
    else:
        o_dict[scene]['N'] += 1
    print('formulating dictionary {}'.format(count))

N = count
N_max = 0
N_min = 99999
min_cls = ''

# find min max of Ns
total = 0
for cls, dic in o_dict.items():
    num = dic['N']
    total += num
    N_max = num if num > N_max else N_max
    N_min, min_cls = (num, cls) if num < N_min else (N_min, min_cls)

delta = 0.01
out_dict = {
    k: {
        'label': v['label'],
        'N': v['N'],
        'frequency': v['N']/total,
        'weight': (v['N']-N_min+delta)/(N_max-N_min)  # no '-1' here !!!!!!!!!!!!!!!!!!!!!
}
    for k, v in o_dict.items()
}

save_dict = out_dict

print('total_Num:', count)
print('min_cls', min_cls)

# save as json
# https://stackoverflow.com/a/7100202
import json
sfile = os.path.join(os.getcwd(), 'sunrgbd_label_weight_dict_Total{}_max{}_min{}.json'.format(N, N_max, N_min))
print('saving dictionary at: '+sfile)
with open(sfile, '+w') as jfile:
    json.dump(save_dict, jfile)

# load json
# with open(sfile, 'r') as jload:
#     dic = json.load(jload)
