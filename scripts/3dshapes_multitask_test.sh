#!/bin/bash


NAME=$2
idx=0

A=(4 4 4 4)

for seed in 1 2 3; do
    export CUDA_VISIBLE_DEVICES="${A[seed]}"
    idx=$((idx+1))
    python3 main.py \
    --name="${NAME}_$idx" \
    --alg=DspritesMT \
    --dset_dir=./data \
    --dset_name=shapes3d_multitask \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --max_epoch 200 \
    --header_dim 64 \
    --header_type=Header \
    --train_output_dir "$1/0/${seed}/train" \
    --test_output_dir "$1/0/${seed}/test" \
    --ckpt_dir "$1/0/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --losses_weights 1 0 0 0 0 0 0 0 0 0\
    --seed ${seed}
done
wait
