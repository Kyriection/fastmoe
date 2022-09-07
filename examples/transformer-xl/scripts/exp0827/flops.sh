echo 'Run training...'
CUDA_VISIBLE_DEVICES=$3 python -u flops_sst2.py \
    --cuda \
    --data ../glue_data/SST-2 \
    --dataset sst2 \
    --n_layer 4 \
    --d_model 256 \
    --n_head 8 \
    --d_head 64 \
    --d_inner $2 \
    --dropout 0.1 \
    --dropatt 0.0 \
    --optim adam \
    --lr 1e-4 \
    --warmup_step 0 \
    --max_step 5000 \
    --eval-interval 500 \
    --log-interval 100 \
    --tgt_len 512 \
    --mem_len 512 \
    --eval_tgt_len 128 \
    --batch_size 16 \
    --work_dir flops \
    --pretrained_weight $1 