#!/bin/bash


NAME=$2
idx=0

A=(7 7 6 6)


echo "name=$NAME"

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
    --header_type Header \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/not-trained/${seed}/train" \
    --test_output_dir "$1/not-trained/${seed}/test" \
    --ckpt_dir "$1/not-trained/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 1 1 1 1 1 1 1 1 1 1 \
    --test True \
    --ckpt_load "$1/not-trained/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
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
    --dset_name="$3" \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type Header \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/multi-10/${seed}/train" \
    --test_output_dir "$1/multi-10/${seed}/test" \
    --ckpt_dir "$1/multi-10/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 1 1 1 1 1 1 1 1 1 1 \
    --test True \
    --ckpt_load "$1/multi-10/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
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
    --dset_name="$3" \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type Header \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/1/${seed}/train" \
    --test_output_dir "$1/1/${seed}/test" \
    --ckpt_dir "$1/1/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 1 0 0 0 0 0 0 0 0 0\
    --test True \
    --ckpt_load "$1/1/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
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
    --dset_name="$3" \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type Header \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/2/${seed}/train" \
    --test_output_dir "$1/2/${seed}/test" \
    --ckpt_dir "$1/2/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 0 1 0 0 0 0 0 0 0 0\
    --test True \
    --ckpt_load "$1/2/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
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
    --dset_name="$3" \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type Header \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/3/${seed}/train" \
    --test_output_dir "$1/3/${seed}/test" \
    --ckpt_dir "$1/3/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 0 0 1 0 0 0 0 0 0 0\
    --test True \
    --ckpt_load "$1/3/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
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
    --dset_name="$3" \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --max_epoch 200 \
    --header_dim 64 \
    --header_type Header \
    --train_output_dir "$1/4/${seed}/train" \
    --test_output_dir "$1/4/${seed}/test" \
    --ckpt_dir "$1/4/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 0 0 0 1 0 0 0 0 0 0\
    --test True \
    --ckpt_load "$1/4/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
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
    --dset_name="$3" \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type Header \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/5/${seed}/train" \
    --test_output_dir "$1/5/${seed}/test" \
    --ckpt_dir "$1/5/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 0 0 0 0 1 0 0 0 0 0\
    --test True \
    --ckpt_load "$1/5/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
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
    --dset_name="$3" \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type Header \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/6/${seed}/train" \
    --test_output_dir "$1/6/${seed}/test" \
    --ckpt_dir "$1/6/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 0 0 0 0 0 1 0 0 0 0\
    --test True \
    --ckpt_load "$1/6/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
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
    --dset_name="$3" \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type Header \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/7/${seed}/train" \
    --test_output_dir "$1/7/${seed}/test" \
    --ckpt_dir "$1/7/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 0 0 0 0 0 0 1 0 0 0\
    --test True \
    --ckpt_load "$1/7/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
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
    --dset_name="$3" \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type Header \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/8/${seed}/train" \
    --test_output_dir "$1/8/${seed}/test" \
    --ckpt_dir "$1/8/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 0 0 0 0 0 0 0 1 0 0\
    --test True \
    --ckpt_load "$1/8/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
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
    --dset_name="$3" \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type Header \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/9/${seed}/train" \
    --test_output_dir "$1/9/${seed}/test" \
    --ckpt_dir "$1/9/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 0 0 0 0 0 0 0 0 1 0\
    --test True \
    --ckpt_load "$1/9/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
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
    --dset_name="$3" \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type Header \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/10/${seed}/train" \
    --test_output_dir "$1/10/${seed}/test" \
    --ckpt_dir "$1/10/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 0 0 0 0 0 0 0 0 0 1\
    --test True \
    --ckpt_load "$1/10/${seed}/ckpt_dir/last" \
    --evaluation_metric factor_vae_metric mig dci irs unsupervised max_corr \
    --seed ${seed} &
done
wait


