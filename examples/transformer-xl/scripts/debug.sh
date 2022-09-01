#!/bin/bash
echo 'Run training...'
python -u train_csqa.py \
    --cuda \
    --data ../data/csqa/ \
    --dataset csqa \
    --n_layer 4 \
    --d_model 256 \
    --n_head 8 \
    --d_head 64 \
    --d_inner 512 \
    --dropout 0.1 \
    --dropatt 0.0 \
    --optim adam \
    --lr $2 \
    --warmup_step 0 \
    --max_step 4000 \
    --eval-interval 500 \
    --log-interval 20 \
    --tgt_len 512 \
    --mem_len 512 \
    --eval_tgt_len 128 \
    --batch_size 1 \
    --moe --moe-num-expert 16 --moe-top-k 2 \
    --gate_name CustomNaiveGate \
    --moe_index 0,1,2,3 \
    --work_dir debug \
    --pretrained_weight $1

