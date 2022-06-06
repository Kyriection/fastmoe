# #! /bin/bash

# # Runs the "345M" parameter model

# RANK=0
# WORLD_SIZE=1

# DATA_PATH=bookcorpus_data_cached/my-gpt_text_document
# CHECKPOINT_PATH=gpt1_book_pretraining


# python pretrain_gpt.py \
#        --num-layers 4 \
#        --hidden-size 192 \
#        --num-attention-heads 3 \
#        --micro-batch-size 4 \
#        --global-batch-size 8 \
#        --seq-length 1024 \
#        --max-position-embeddings 1024 \
#        --train-iters 500000 \
#        --lr-decay-iters 320000 \
#        --save $CHECKPOINT_PATH \
#        --load $CHECKPOINT_PATH \
#        --data-path $DATA_PATH \
#        --vocab-file gpt2-vocab.json \
#        --merge-file gpt2-merges.txt \
#        --data-impl mmap \
#        --split 949,50,1 \
#        --distributed-backend nccl \
#        --lr 0.00015 \
#        --min-lr 1.0e-5 \
#        --lr-decay-style cosine \
#        --weight-decay 1e-2 \
#        --clip-grad 1.0 \
#        --lr-warmup-fraction .01 \
#        --log-interval 100 \
#        --save-interval 10000 \
#        --eval-interval 1000 \
#        --eval-iters 10 \
#        --fp16





#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

DATA_PATH=bookcorpus_data_cached/my-gpt_text_document
CHECKPOINT_PATH=gpt1_moe_book_pretraining


python pretrain_gpt.py \
       --num-layers 2 \
       --hidden-size 96 \
       --num-attention-heads 3 \
       --micro-batch-size 4 \
       --global-batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file gpt2-vocab.json \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --fmoefy \
       --num-experts-moe 4 \
       --top-k 2 