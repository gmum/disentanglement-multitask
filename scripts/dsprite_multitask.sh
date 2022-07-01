#!/bin/bash

FILENAME=$(basename $0)
FILENAME="${FILENAME%.*}"
NAME=${1:-$FILENAME}

echo "name=$NAME"

python3 main.py \
--name=$NAME \
--alg=DspritesMT \
--dset_dir=./data \
--dset_name=dsprites_multitask \
--encoder=SimpleConv64 \
--decoder=SimpleConv64 \
--z_dim=8 \
--max_epoch 2 \
--header_dim 64 \
--train_output_dir "$1/train" \
--test_output_dir "$1/test" \
--ckpt_dir "$1/ckpt_dir" \
--batch_size 256 \
--neptune-logging \
--evaluate_iter 1 \
--evaluation_metric irs factor_vae_metric mig unsupervised \
--losses_weights 1 0 0 0 0 \
--seed 123 

