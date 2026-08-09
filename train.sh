export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py dct_eigen_pff_phase4 --gpus 6
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py dct_eigen_pff_phase4 --gpus 6 --pseudo_weight 0 --save_dir checkpoints_meanteacher_0 #&
#CUDA_VISIBLE_DEVICES=1 python train.py dct_eigen_pff_phase4 --gpus 1 --pseudo_weight 0.005 --save_dir checkpoints_meanteacher_0.005 &
#CUDA_VISIBLE_DEVICES=2 python train.py dct_eigen_pff_phase4 --gpus 1 --pseudo_weight 0.01 --save_dir checkpoints_meanteacher_0.01 &
#CUDA_VISIBLE_DEVICES=3 python train.py dct_eigen_pff_phase4 --gpus 1 --pseudo_weight 0.015 --save_dir checkpoints_meanteacher_0.015 &
#CUDA_VISIBLE_DEVICES=4 python train.py dct_eigen_pff_phase4 --gpus 1 --pseudo_weight 0.02 --save_dir checkpoints_meanteacher_0.02 &
#CUDA_VISIBLE_DEVICES=5 python train.py dct_eigen_pff_phase4 --gpus 1 --pseudo_weight 0.025 --save_dir checkpoints_meanteacher_0.025 &
#wait
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py dct_eigen_pff_phase4 --gpus 6 
#tensorboard --logdir=checkpoints_phase3/dct_eigen_pff_phase3/dcdepth_logs/ --port=6036 &
#tensorboard --logdir=checkpoints_baseline/dct_eigen_pff_phase3/dcdepth_logs/ --port=6049 &
#tensorboard --logdir=checkpoints_fixmatch/dct_eigen_pff_phase4/dcdepth_logs/ --port=6099 &
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train.py dct_eigen_pff_phase4 --gpus 6
CUDA_VISIBLE_DEVICES=0,2,3 python train.py tcd_nyu_pff_phase4 --gpus 3