#!/bin/bash
echo 'Run training...'
python -u train.py \
    --cuda \
    --data ../data/enwik8/ \
    --dataset enwik8 \
    --n_layer 4 \
    --d_model 256 \
    --n_head 8 \
    --d_head 64 \
    --d_inner 2048 \
    --dropout 0.1 \
    --dropatt 0.0 \
    --optim adam \
    --lr 0.00025 \
    --warmup_step 0 \
    --max_step 20000 \
    --tgt_len 512 \
    --mem_len 512 \
    --eval_tgt_len 128 \
    --batch_size 88 \
    --moe --moe-num-expert 4 --moe-top-k 2 \
    --gate_name CustomNaiveGate \
    --freeze_gate \
    --dynamic_moe \
    --dynamic_moe_mode linear_increase \
    --dynamic_overall_steps 20000 \
    --moe-top-k-min 2 \
    --moe-top-k-max 4 \
    --work_dir EXPERTS4_Test
