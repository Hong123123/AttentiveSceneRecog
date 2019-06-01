# AttentiveSceneRecog
scene recognition with attention

## data preprocessing:
1) use <b>utility/make_hha_encode/encode_sun_rgbd.py</b> to generate hha image, which is time costly.

2) use <b>nyud2_meta/make_common_10.py</b> to generate meta data of the dataset;

## run train:
python work.py --lr 3e-4 --l2 0.5 --log_root /path/to/tensorboard_log/ --epochs 10000 --pretrain_dir /path/to/alexnet_places365.pth.tar --save_inteval_epoch 1 --device cuda:0
