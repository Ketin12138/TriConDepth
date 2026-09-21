export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0,2,3 python train.py tcd_nyu_pff_phase4 --gpus 3
