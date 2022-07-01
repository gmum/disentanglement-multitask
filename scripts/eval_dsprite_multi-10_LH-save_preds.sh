#!/bin/bash


NAME=$2
idx=0

A=(2 2 2 2)


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
    --header_type DatasetHeader \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/10/${seed}/train" \
    --test_output_dir "$1/10/${seed}/test" \
    --ckpt_dir "$1/10/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 1 1 1 1 1 1 1 1 1 1 \
    --test True \
    --save-preds True \
    --ckpt_load "$1/10/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
    --seed ${seed} &
done
wait


