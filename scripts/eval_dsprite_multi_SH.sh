#!/bin/bash


NAME=$2
idx=0

A=(0 0 2 2)

echo "name=$NAME"
for seed in 1 2 3; do
    export CUDA_VISIBLE_DEVICES="${A[seed]}"
    idx=$((idx+1))
    python3 main.py \
    --name="$NAME_${idx}" \
    --alg=DspritesMT \
    --dset_dir=./data \
    --dset_name=dsprites_multitask \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type Header \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/multi-5/${seed}/train" \
    --test_output_dir "$1/multi-5/${seed}/test" \
    --ckpt_dir "$1/multi-5/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --test True \
    --ckpt_load "$1/multi-5/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
    --losses_weights 1 1 1 1 1 1 1 1 1 1\
    --seed ${seed} &
done

for seed in 1 2 3; do
    export CUDA_VISIBLE_DEVICES="${A[seed]}"
    idx=$((idx+1))
    python3 main.py \
    --name="${NAME}_${idx}" \
    --alg=DspritesMT \
    --dset_dir=./data \
    --dset_name=dsprites_multitask \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type Header \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/multi-20/${seed}/train" \
    --test_output_dir "$1/multi-20/${seed}/test" \
    --ckpt_dir "$1/multi-20/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 20 \
    --test True \
    --ckpt_load "$1/multi-20/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
    --losses_weights 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\
    --seed ${seed} &
done
wait


for seed in 1 2 3; do
    export CUDA_VISIBLE_DEVICES="${A[seed]}"
    idx=$((idx+1))
    python3 main.py \
    --name="${NAME}_${idx}" \
    --alg=DspritesMT \
    --dset_dir=./data \
    --dset_name=dsprites_multitask \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type Header \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/multi-30/${seed}/train" \
    --test_output_dir "$1/multi-30/${seed}/test" \
    --ckpt_dir "$1/multi-30/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 30 \
    --test True \
    --ckpt_load "$1/multi-30/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
    --losses_weights 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\
    --seed ${seed} &
done

for seed in 1 2 3; do
    export CUDA_VISIBLE_DEVICES="${A[seed]}"
    idx=$((idx+1))
    python3 main.py \
    --name="${NAME}_${idx}" \
    --alg=DspritesMT \
    --dset_dir=./data \
    --dset_name=dsprites_multitask \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type Header \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/multi-40/${seed}/train" \
    --test_output_dir "$1/multi-40/${seed}/test" \
    --ckpt_dir "$1/multi-40/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 40 \
    --test True \
    --ckpt_load "$1/multi-40/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
    --losses_weights 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\
    --seed ${seed} &
done
wait

