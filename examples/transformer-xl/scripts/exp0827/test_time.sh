# #!/bin/bash
# python -u train_block.py \
#     --cuda \
#     --data ../data/enwik8/ \
#     --dataset enwik8 \
#     --n_layer 4 \
#     --d_model 256 \
#     --n_head 8 \
#     --d_head 64 \
#     --d_inner 8192 \
#     --dropout 0.1 \
#     --dropatt 0.0 \
#     --optim adam \
#     --lr 0.00025 \
#     --warmup_step 0 \
#     --max_step 107317 \
#     --tgt_len 512 \
#     --mem_len 512 \
#     --eval_tgt_len 128 \
#     --batch_size 22 \
#     --work_dir big_dense_block 


# python -u train.py \
#     --cuda \
#     --data ../data/enwik8/ \
#     --dataset enwik8 \
#     --n_layer 4 \
#     --d_model 256 \
#     --n_head 8 \
#     --d_head 64 \
#     --d_inner 8192 \
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
#     --work_dir big_dense 


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
    --max_step 100000 \
    --tgt_len 512 \
    --mem_len 512 \
    --eval_tgt_len 128 \
    --batch_size 22 \
    --moe --moe-num-expert 16 --moe-top-k 2 \
    --gate_name CustomRandomGate \
    --work_dir THOR_4layer_16experts \
    --kl_alpha 2 
