#! bin/bash
export CUDA_VISIBLE_DEVICES=0
DEVICE=0
EXP_NAME=1Task-NUT-ASSEMBLY
# name of the task to use for training
TASK_str=nut_assembly
EPOCH=1
BSIZE=12
python3 ../train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=${EXP_NAME} bsize=${BSIZE} actions.n_mixtures=2 actions.out_dim=64 attn.attn_ff=128  simclr.mul_intm=0 simclr.compressor_dim=128 simclr.hidden_dim=256 epochs=${EPOCH} device=${DEVICE} 