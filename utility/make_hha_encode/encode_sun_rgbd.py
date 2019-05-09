# two reference !
# https://github.com/ZhangMenghe/rgbd-processor-python
# stacypinks.com/github_/topics/nyuv2
import numpy as np
import imageio
import math
import os
from os import walk
from utility.make_hha_encode.rgbd_processor_python.camera import processCamMat
from utility.make_hha_encode.rgbd_processor_python.depthImgProcessor import processDepthImage
import config
import glob
import sys


sys.path.append('/mnt/old_hexin/AttentiveScenery/utility/make_hha_encode/rgbd_processor_python')


def getHHAImg(depthImage, missingMask, cameraMatrix):
    pc, N, yDir, h, R = processDepthImage(depthImage * 100, missingMask, cameraMatrix)

    tmp = np.multiply(N, yDir)
    acosValue = np.minimum(1, np.maximum(-1,np.sum(tmp, axis=2)))
    angle = np.array([math.degrees(math.acos(x)) for x in acosValue.flatten()])
    angle = np.reshape(angle, h.shape)

    '''
        Must convert nan to 180 as the MATLAB program actually does. 
        Or we will get a HHA image whose border region is different
        with that of MATLAB program's output.
        '''
    angle[np.isnan(angle)] = 180

    pc[:,:,2] = np.maximum(pc[:,:,2], 100)
    I = np.zeros(pc.shape)
    I[:,:,0] = 31000/pc[:,:,2]
    I[:,:,1] = h
    I[:,:,2] = (angle + 128-90)

    '''
       np.uint8 seems to use 'floor', but in matlab, it seems to use 'round'.
       So I convert it to integer myself.
    '''
    I = np.rint(I)

    '''
        # np.uint8: 256->1, but in MATLAB, uint8: 256->255
    '''
    I[I > 255] = 255

    HHA = I.astype(np.uint8)
    return HHA


def main():
    root_path = config.sunrgbd_root
    # sun_path = root_path + '/SUNRGBD_HHA'
    sun_path = '/mnt/pub_workspace_2T/hong_data/SUNRGBD_root/SUNRGBD_HHA/kv1/NYUdata'
    target = 'scene.txt'

    count = 0
    for base_dir, _, filelist in walk(sun_path):
        for file in filelist:
            if file == target:
                camAddr = base_dir + '/intrinsics.txt'
                with open(camAddr, 'r') as camf:
                    cameraMatrix = processCamMat(camf.readlines())

                depthAddr = glob.glob(base_dir + '/depth_bfx/' + '*.png')[0]
                depth_name = depthAddr.split('/')[-1]
                rawDepthAddr = glob.glob(base_dir + '/depth/' + '*.png')[0]

                depthImage = imageio.imread(depthAddr).astype(float)/10000
                rawDepth = imageio.imread(rawDepthAddr).astype(float)/100000
                missingMask = (rawDepth == 0)

                HHA = getHHAImg(depthImage, missingMask, cameraMatrix)

                hha_folder = '/hha_2'
                height_folder = '/height_2'

                hha_dir = base_dir + hha_folder
                height_dir = base_dir + height_folder
                os.makedirs(hha_dir) if not os.path.exists(hha_dir) else ''
                os.makedirs(height_dir) if not os.path.exists(height_dir) else ''

                imageio.imwrite(base_dir + hha_folder+ '/' + depth_name + '.png', HHA)
                imageio.imwrite(base_dir + height_folder + '/' + depth_name + '.png', HHA[:,:,1])
                count += 1
                print('processing {}-th image'.format(count))
                print('at {}'.format(base_dir + hha_folder+ '/' + depth_name + '.png'))


if __name__ == "__main__":
    main()
    print('-'*5+'finished'+'-'*5)