#! /bin/bash
export CUDA_VISIBLE_DEVICES=2
python3 -u train_demo.py \
    --trainN 10 \
    --N 10 \
    --K 5 \
    --Q 1 \
    --model HND \
    --encoder bert \
    --hidden_size 768 \
    --val_step 1000 \
    --batch_size 2  \
  | tee logs/1099.train.log 2>&1
