#!/bin/bash
echo 'Run training...'
python -u train.py \
    --cuda \
    --data ../data/enwik8/ \
    --dataset enwik8 \
    --n_layer $1 \
    --d_model 256 \
    --n_head 8 \
    --d_head 64 \
    --d_inner $2 \
    --dropout 0.1 \
    --dropatt 0.0 \
    --optim adam \
    --lr 0.00025 \
    --warmup_step 0 \
    --max_step 400000 \
    --tgt_len 512 \
    --mem_len 512 \
    --eval_tgt_len 128 \
    --batch_size 22 \
    --work_dir $3 \
    --multi_gpu \
    --gpu0_bsz 4
