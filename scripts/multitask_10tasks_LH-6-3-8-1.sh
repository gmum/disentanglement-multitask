#!/bin/bash


NAME=$2
idx=0

A=(0 1 6 1)


for seed in 1 2 3; do
    export CUDA_VISIBLE_DEVICES="${A[seed]}"
    idx=$((idx+1))
    python3 main.py \
    --name="${NAME}_${idx}" \
    --alg=DspritesMT \
    --dset_dir=./data \
    --dset_name="$3" \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type DatasetHeader \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/6/${seed}/train" \
    --test_output_dir "$1/6/${seed}/test" \
    --ckpt_dir "$1/6/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 0 0 0 0 0 1 0 0 0 0\
    --seed ${seed} &
done


for seed in 1; do
    export CUDA_VISIBLE_DEVICES="${A[seed]}"
    idx=$((idx+1))
    python3 main.py \
    --name="${NAME}_${idx}" \
    --alg=DspritesMT \
    --dset_dir=./data \
    --dset_name="$3" \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type DatasetHeader \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/8/${seed}/train" \
    --test_output_dir "$1/8/${seed}/test" \
    --ckpt_dir "$1/8/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 0 0 0 0 0 0 0 1 0 0\
    --seed ${seed} &
done
wait




