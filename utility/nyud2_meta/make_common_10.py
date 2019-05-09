import h5py
import json
import numpy as np

import config
'''
sample_sceneTypes
sample_sceneIndices
scene_names
scene_rat_10
scene_map
'''


def save_json(obj, save_dir):
    with open(save_dir, '+w') as file:
        json.dump(obj, file)


def load_json(load_dir):
    with open(load_dir, 'r') as file:
        obj = json.load(file)
    return obj


# IO paths
# >>>>>>
save_folder = config.root + '/utility/nyud2_meta'
sample_sceneTypes_dir = save_folder + '/sample_sceneTypes.json'
sample_sceneIndices_dir = save_folder + '/sample_sceneIndices.json'
scene_names_dir = save_folder + '/scene_names.json'
scene_rat_10_dir = save_folder + '/scene_rat_10.json'
scene_map_dir = save_folder + '/scene_map.json'
# <<<<<<

if __name__ == '__main__':  # generate meta data
    mat_dir = '/mnt/pub_workspace_2T/hong_data/NYUv2_ROOT/nyu_depth_v2_labeled.mat'

    f = h5py.File(mat_dir)
    s = f['scenes']
    obj_refs = s[0, :]

    sample_scenes = [''.join([chr(i) for i in f[r]]) for r in obj_refs]

    sample_sceneTypes = np.array([s[:s.rfind('_')] for s in sample_scenes])
    save_json(sample_sceneTypes.tolist(), sample_sceneTypes_dir)

    # columns for cls_idx: scene_names  scene_fres  map_source  scene_map
    scene_names, sample_sceneIndices = np.unique(sample_sceneTypes, return_inverse=True)
    assert np.all(scene_names[sample_sceneIndices] == sample_sceneTypes)
    save_json(scene_names.tolist(), scene_names_dir)
    save_json(sample_sceneIndices.tolist(), sample_sceneIndices_dir)

    # calculate cls frequency
    scene_fres = np.zeros_like(scene_names, dtype=np.int)
    for idx in sample_sceneIndices:
        scene_fres[idx] += 1

    common9 = np.argsort(scene_fres)[-1:-10:-1]  # desc
    others = np.array([idx for idx in range(len(scene_names)) if idx not in common9])

    scene_common9 = {str(source): i for i, source in enumerate(common9)}  # map to its pos (0~8) in common9
    scene_others = {str(source): 9 for source in others}  # map to pos 9

    # map_source = dict(**scene_common9, **scene_others)

    scene_map = [
        (scene_common9 if str(idx) in scene_common9.keys() else scene_others)[str(idx)]
        for idx in range(len(scene_names))
    ]
    save_json(scene_map, scene_map_dir)

    # calculate cls frequency with common 10
    scene_fres_10 = np.zeros((10,), dtype=np.int)
    for idx in sample_sceneIndices:
        scene_fres_10[scene_map[idx]] += 1

    scene_rat_10 = scene_fres_10 / np.sum(scene_fres_10)
    save_json(scene_rat_10.tolist(), scene_rat_10_dir)

    print('fres with common 10:', scene_fres_10)
    print('fres with common 9:', scene_fres[common9])
    print('names with common 9:', scene_names[common9])
    print('names of others:', others)

# load meta data
else:  # __name__ != '__main__':
    sample_sceneTypes = load_json(sample_sceneTypes_dir)
    sample_sceneIndices = load_json(sample_sceneIndices_dir)
    scene_names = load_json(scene_names_dir)
    scene_rat_10 = load_json(scene_rat_10_dir)
    scene_map = load_json(scene_map_dir)
