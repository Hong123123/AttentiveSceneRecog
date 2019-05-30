# train with params
python refractored_train.py --lr 3e-8 --log_root '/mnt/old_hexin/log_atten' --epochs 1000 --ckpt_dir '/mnt/old_hexin/log_atten/2019-04-27_13:18:27/checkpoints/best/epochs:102.ckpt' --save_inteval_epoch 10 --device 'cuda:2'

# legacy0
python refractored_train.py --lr 3e-4 --log_root '/mnt/old_hexin/log_atten' --epochs 100 --ckpt_dir '/mnt/old_hexin/log/2019-04-26_20:42:46/checkpoints/best/epochs:92.ckpt' --eval_inteval_epoch 1 --save_inteval_epoch 1

# legacy1
python refractored_train.py --lr 1e-3 --log_root '/mnt/old_hexin/log_atten' --epochs 100 --ckpt_dir '/mnt/old_hexin/log/2019-04-26_20:42:46/checkpoints/best/epochs:92.ckpt' --eval_inteval_epoch 1 --save_inteval_epoch 1

# train with no atten: from scratch
python refractored_train.py --lr 1e-3 --log_root '/mnt/old_hexin/log_atten/new' --epochs 1000 --pretrain_dir '/mnt/pub_workspace_2T/hong_data/AttentiveScenery/models/ckpt/alexnet_places365.pth.tar' --save_inteval_epoch 10 --device 'cuda:1'

# train with no atten: resume
python refractored_train.py --no_atten true --lr 1e-3 --log_root /mnt/old_hexin/log_atten/no_atten --epochs 1000 --ckpt_dir /mnt/old_hexin/log_atten/new/test/2019-04-27_21:22:38/checkpoints/best/epochs:4.ckpt --save_inteval_epoch 10 --device cuda:0

# train with no atten: scrath, all
python refractored_train.py --lr 1e-3 --split all --log_root '/mnt/old_hexin/log_atten/all_scratch' --epochs 1000 --pretrain_dir '/mnt/pub_workspace_2T/hong_data/AttentiveScenery/models/ckpt/alexnet_places365.pth.tar' --save_inteval_epoch 10 --device 'cuda:2' --no_atten true

## debug
# overfit from init
--no_atten true --lr 1e-3 --log_root /mnt/old_hexin/log_atten/debug --epochs 1000 --pretrain_dir /mnt/pub_workspace_2T/hong_data/AttentiveScenery/models/ckpt/alexnet_places365.pth.tar --save_inteval_epoch 10 --device cuda:0

best ckpt:
/mnt/old_hexin/log_atten/debug/2019-05-01_14:15:23/checkpoints/best/val_acc_of_0.95_at_epoch:102.ckpt

# resume from initial overfit
--no_atten true --lr 3e-4 --log_root /mnt/old_hexin/log_atten/init_well --epochs 500 --ckpt_dir /mnt/old_hexin/log_atten/debug/2019-05-01_17:13:25/checkpoints/best/val_acc_of_0.975_at_epoch:106.ckpt --save_inteval_epoch 10 --device cuda:2

## work from scrath no atten
--no_atten true --lr 3e-4 --log_root /mnt/old_hexin/log_atten/init_well --epochs 500 --pretrain_dir /mnt/pub_workspace_2T/hong_data/AttentiveScenery/models/ckpt/alexnet_places365.pth.tar --save_inteval_epoch 10 --device cuda:2

# work from scratch
--no_atten false --lr 1 --log_root /mnt/old_hexin/log_atten/init_well --epochs 500 --pretrain_dir /mnt/pub_workspace_2T/hong_data/AttentiveScenery/models/ckpt/alexnet_places365.pth.tar --save_inteval_epoch 10 --device cuda:1

# work big_LR
--no_atten false --lr 1 --l2 2 --log_root /mnt/old_hexin/log_atten/big_LR --epochs 500 --ckpt_dir /mnt/old_hexin/log_atten/big_LR/2019-05-01_20:09:43/checkpoints/val_acc_of_0.050019091256204656_at_epoch:8.ckpt --save_inteval_epoch 10 --device cuda:1

# work big_crazy
--no_atten false --lr 1 --l2 1 --log_root /mnt/old_hexin/log_atten/big_LR --epochs 500 --ckpt_dir /mnt/old_hexin/log_atten/big_LR/2019-05-01_20:09:43/checkpoints/val_acc_of_0.050019091256204656_at_epoch:8.ckpt --save_inteval_epoch 10 --device cuda:0

# work from big_LR
--no_atten false --lr 3e-4 --l2 2 --log_root /mnt/old_hexin/log_atten/big_LR --epochs 500 --ckpt_dir /mnt/old_hexin/log_atten/big_LR/2019-05-01_20:09:43/checkpoints/val_acc_of_0.050019091256204656_at_epoch:8.ckpt --save_inteval_epoch 10 --device cuda:1

## fine tuned
# small LR
--no_atten false --lr 1e-8 --l2 0.5 --log_root /mnt/old_hexin/log_atten/fine --epochs 500 --pretrain_dir /mnt/pub_workspace_2T/hong_data/AttentiveScenery/models/ckpt/alexnet_places365.pth.tar --save_inteval_epoch 2 --device cuda:1
# normal LR
--no_atten false --lr 3e-4 --l2 0.5 --log_root /mnt/old_hexin/log_atten/fine --epochs 500 --pretrain_dir /mnt/pub_workspace_2T/hong_data/AttentiveScenery/models/ckpt/alexnet_places365.pth.tar --save_inteval_epoch 2 --device cuda:2

## nyud v2
--no_atten true --lr 3e-4 --l2 0.5 --log_root /home/hong/small_log_atten/nyudv2 --epochs 10000 --pretrain_dir /mnt/pub_workspace_2T/hong_data/AttentiveScenery/models/ckpt/alexnet_places365.pth.tar --save_inteval_epoch 1 --device cuda:1

--no_atten false --lr 3e-4 --l2 0.5 --log_root /home/hong/small_log_atten/nyudv2 --epochs 10000 --pretrain_dir /mnt/pub_workspace_2T/hong_data/AttentiveScenery/models/ckpt/alexnet_places365.pth.tar --save_inteval_epoch 1 --device cuda:2

# startup baseline
python work_full_adam_baseline.py --lr 3e-4 --l2 0.5 --log_root /home/hong/small_log_atten/nyudv2/baseline --epochs 10000 --pretrain_dir /mnt/pub_workspace_2T/hong_data/AttentiveScenery/models/ckpt/alexnet_places365.pth.tar --save_inteval_epoch 1 --device cuda:0

# >>>>
# try fine-tune, from 0.54
python work_full_freeze_baseline.py --lr 1e-30 --l2 0.5 --log_root /home/hong/small_log_atten/nyudv2/freeze_baseline --epochs 100000 --ckpt_dir /home/hong/small_log_atten/nyudv2/freeze_baseline/origin_2019-05-09_22:56:27/checkpoints/val_acc_of_0.5412844036697247_at_epoch:98.ckpt --save_inteval_epoch 1 --device cuda:2

# <<<<<

# with atten
python work_full_adam.py --no_atten false --lr 3e-4 --l2 10 --log_root /home/hong/small_log_atten/nyudv2/has_atten --epochs 10000 --ckpt_dir /home/hong/small_log_atten/nyudv2/has_atten/2019-05-09_15:58:13/checkpoints/best/val_acc_of_0.43425076452599387_at_epoch:6872.ckpt --save_inteval_epoch 1 --device cuda:1

# a baseline
# >>> file: work_full_adam_baseline.py
python work_full_adam_baseline.py --lr 3e-4 --l2 0.5 --log_root /home/hong/small_log_atten/nyudv2/baseline --epochs 10000 --ckpt_dir /home/hong/small_log_atten/nyudv2/baseline/2019-05-07_21:14:04/checkpoints/val_acc_of_0.3654434250764526_at_epoch:95.ckpt --save_inteval_epoch 1 --device cuda:0

# <<<<<<<<<<<<<<<<<

# >>>> running >>>>
# freeze baseline, origin
python work_full_freeze_baseline.py --lr 3e-4 --l2 0.1 --log_root /home/hong/small_log_atten/nyudv2/freeze_baseline/debug --epochs 100000 --pretrain_dir /mnt/pub_workspace_2T/hong_data/AttentiveScenery/models/ckpt/alexnet_places365.pth.tar --save_inteval_epoch 1 --device cuda:2

/home/hong/small_log_atten/nyudv2/freeze_baseline/origin_2019-05-09_22:56:27/checkpoints/best/val_acc_of_0.5871559633027523_at_epoch:30.ckpt

# try fine-tune, from 0.59
python work_full_freeze_baseline.py --lr 1e-30 --l2 0.1 --log_root /home/hong/small_log_atten/nyudv2/freeze_baseline --epochs 100000 --ckpt_dir /home/hong/small_log_atten/nyudv2/freeze_baseline/val_acc_of_0.5902140672782875_at_epoch:51.ckpt --save_inteval_epoch 1 --device cuda:2

# fine-tunning! good, LR:1k~10k
python work_full_freeze_baseline.py --lr 1e-10000 --l2 0.1 --log_root /home/hong/small_log_atten/nyudv2/freeze_baseline --epochs 100000 --ckpt_dir /home/hong/small_log_atten/nyudv2/freeze_baseline/val_acc_of_0.5978593272171254_at_epoch:111.ckpt --save_inteval_epoch 1 --device cuda:1

# aggressive hageshii, useless
python work_full_adam_baseline.py --lr 1e-100000000000000 --l2 0.1 --log_root /home/hong/small_log_atten/nyudv2/freeze_baseline --epochs 100000 --ckpt_dir /home/hong/small_log_atten/nyudv2/freeze_baseline/val_acc_of_0.5978593272171254_at_epoch:111.ckpt --save_inteval_epoch 1 --device cuda:0

