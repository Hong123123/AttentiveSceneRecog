import os
from collections import OrderedDict

rootDir = '/mnt/pub_workspace_2T/hong_data/SUNRGBD/'
target = 'scene.txt'
o_dict = OrderedDict()

## calculate percentage
print('formulating percentage')
num = 0
for _,_,fileList in os.walk(rootDir):
    for file_name in fileList:
        if file_name == target:
            num += 1

index = 0
count = 0
for base_dir,_,fileList in os.walk(rootDir):
    for file_name in fileList:
        if file_name == target:
            count += 1
            with open(os.path.join(base_dir,file_name), 'r') as file:
                scene=file.read()
            if scene not in o_dict.keys():
                o_dict[scene] = index
                index += 1
                print('formulating dictionary {:.2%}'.format(count/num))

# save as json
# https://stackoverflow.com/a/7100202
import json
sfile = os.path.join(os.getcwd(),'sunrgbd_label_dict.json')
print('saving dictionary at: '+sfile)
with open(sfile, '+w') as jfile:
    json.dump(o_dict, jfile)

# load json
# with open(sfile, 'r') as jload:
#     dic = json.load(jload)
