#!/bin/bash
echo 'Run training...'
python -u train_thor.py \
    --cuda \
    --data ../data/enwik8/ \
    --dataset enwik8 \
    --n_layer 4 \
    --d_model 256 \
    --n_head 8 \
    --d_head 64 \
    --d_inner 512 \
    --dropout 0.1 \
    --dropatt 0.0 \
    --optim adam \
    --lr 0.00025 \
    --warmup_step 0 \
    --max_step 100000
    --tgt_len 512 \
    --mem_len 512 \
    --eval_tgt_len 128 \
    --batch_size 88 \
    --multi_gpu \
    --gpu0_bsz 11 \
    --moe --moe-num-expert 16 --moe-top-k 2 \
    --gate_name CustomRandomGate \
    --work_dir THOR_4layer_16experts \
    --kl_alpha 2 
