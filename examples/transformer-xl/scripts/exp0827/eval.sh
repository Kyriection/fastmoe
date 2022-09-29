# #!/bin/bash
# echo 'Run Evaluation...'
# CUDA_VISIBLE_DEVICES=$3 nohup python -u evaluation.py \
#     --cuda \
#     --data ../data/enwik8/ \
#     --dataset enwik8 \
#     --n_layer 4 \
#     --d_model 256 \
#     --n_head 8 \
#     --d_head 64 \
#     --d_inner $1 \
#     --dropout 0.1 \
#     --dropatt 0.0 \
#     --optim adam \
#     --lr 0.00025 \
#     --warmup_step 0 \
#     --max_step 400000 \
#     --tgt_len 512 \
#     --mem_len 512 \
#     --eval_tgt_len 128 \
#     --batch_size 22 \
#     --checkpoint_weight $2 > log_eval_dense_dim$1.out 2>&1 &






echo 'Run Evaluation...'
CUDA_VISIBLE_DEVICES=$2 nohup python -u evaluation.py \
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
    --max_step 400000 \
    --tgt_len 512 \
    --mem_len 512 \
    --eval_tgt_len 128 \
    --batch_size 22 \
    --moe --moe-num-expert 16 --moe-top-k 2 \
    --gate_name CustomHashGate \
    --checkpoint_weight $1 > log_eval_$3.out 2>&1 &



# echo 'Run Evaluation...'
# CUDA_VISIBLE_DEVICES=$2 nohup python -u evaluation.py \
#     --cuda \
#     --data ../data/enwik8/ \
#     --dataset enwik8 \
#     --n_layer 4 \
#     --d_model 256 \
#     --n_head 8 \
#     --d_head 64 \
#     --d_inner 512 \
#     --dropout 0.1 \
#     --dropatt 0.0 \
#     --optim adam \
#     --lr 0.00025 \
#     --warmup_step 0 \
#     --max_step 400000 \
#     --tgt_len 512 \
#     --mem_len 512 \
#     --eval_tgt_len 128 \
#     --batch_size 22 \
#     --moe --moe-num-expert 16 --moe-top-k 2 \
#     --gate_name CustomRandomGate \
#     --checkpoint_weight $1 > log_eval_$3.out 2>&1 &
