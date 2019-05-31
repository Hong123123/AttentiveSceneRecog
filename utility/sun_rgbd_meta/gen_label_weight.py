import os
from collections import OrderedDict
import config

# rootDir = config.sunrgbd_root
rootDir = '/mnt/pub_workspace_2T/hong_data/SUNRGBD_root/SUNRGBD/'
target = 'seg.mat'
sceneName = 'scene.txt'
o_dict = OrderedDict()

## calculate percentage
# print('formulating percentage')
# num = 0
# for _,_,fileList in os.walk(rootDir):
#     for file_name in fileList:
#         if file_name == target:
#             num += 1

index = 0
count = 0
for base_dir,_,fileList in os.walk(rootDir):
    for file_name in fileList:
        if file_name == sceneName:
            count += 1
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
for cls, dic in o_dict.items():
    num = dic['N']
    N_max = num if num > N_max else N_max
    N_min, min_cls = (num, cls) if num < N_min else (N_min, min_cls)

delta = 0.01
out_dict = {k:{'label': v['label'], 'N':v['N'], 'frequency': (v['N']-N_min+delta)/(N_max-N_min)} for k,v in o_dict.items()}  # typo: 'frequency' should be 'weight'

save_dict = out_dict

print('total_Num:', count)
print('min_cls', min_cls)

# save as json
# https://stackoverflow.com/a/7100202
import json
sfile = os.path.join(os.getcwd(),'sunrgbd_label_fre_dict_{}_{}_{}.json'.format(N, N_max, N_min))
print('saving dictionary at: '+sfile)
with open(sfile, '+w') as jfile:
    json.dump(save_dict, jfile)

# load json
# with open(sfile, 'r') as jload:
#     dic = json.load(jload)
