import os
from collections import OrderedDict
import shutil
from shutil import copyfile

rootDir = '/mnt/pub_workspace_2T/hong_data/SUNRGBD/SUNRGBD/'
destination = '/mnt/pub_workspace_2T/hong_data/SUNRGBD/SUNRGBD_re/'
# rootDir = '/mnt/pub_workspace_2T/hong_data/SUNRGBD/utils/test_db/'
# destination = '/mnt/pub_workspace_2T/hong_data/SUNRGBD/utils/test_gen_db/'
target = 'scene.txt'

## calculate percentage
print('formulating percentage')
num = 0
for _,_,fileList in os.walk(rootDir):
    for file_name in fileList:
        if file_name == target:
            num += 1

index = 0
# structure /destination/data_index/[label, label_index, RGB, Depth]
for base_dir,_,fileList in os.walk(rootDir):
    for file_name in fileList:
        if file_name == target:
            fullname = os.path.join(base_dir,file_name)
            fulltarget = fullname
            with open(fullname, 'r') as file:
                label=file.read()
            current_des = os.path.join(destination, str(index))
            if not os.path.exists(current_des):
                os.makedirs(current_des)

            # copy RBG
            shutil.copytree(os.path.join(base_dir,'image'),os.path.join(current_des, 'image'))
            # copy depth
            shutil.copytree(os.path.join(base_dir, 'depth'), os.path.join(current_des, 'depth'))
            # copy label
            copyfile(fulltarget, os.path.join(current_des, target))

            index += 1
            print('processing: {:.2%}'.format(index/num))
            break

