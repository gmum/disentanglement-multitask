#!/bin/bash

#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=10
#SBATCH --cpus-per-gpu=10

source activate py3pytorch1.4

NAME=$2
idx=0

A=(0 0 0 0)

echo "name=$NAME"
for seed in 1 2 3; do
    echo "${A[seed]}"
    export CUDA_VISIBLE_DEVICES="${A[seed]}"
    idx=$((idx+1))
    python3 main.py \
    --name="${NAME}_${idx}" \
    --alg=FullRegression \
    --dset_dir=./data \
    --dset_name="$3" \
    --encoder=SimpleConv64 \
    --decoder=SimpleConv64 \
    --z_dim=8 \
    --header_type DatasetHeader \
    --max_epoch 200 \
    --header_dim 64 \
    --train_output_dir "$1/regression/${seed}/train" \
    --test_output_dir "$1/regression/${seed}/test" \
    --ckpt_dir "$1/regression/${seed}/ckpt_dir" \
    --batch_size 256 \
    --neptune-logging \
    --n_task_headers 10 \
    --losses_weights 1 1 1 1 1 1 1 1 1 1 \
    --test True \
    --ckpt_load "$1/regression/${seed}/ckpt_dir/last" \
    --evaluation_metric sap_score factor_vae_metric mig dci irs unsupervised max_corr \
    --seed ${seed} 
done
wait
#sleep 180





