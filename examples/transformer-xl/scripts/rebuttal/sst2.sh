echo 'Run training...'
CUDA_VISIBLE_DEVICES=$3 nohup python -u train_sst2.py \
    --cuda \
    --data ../glue_data/SST-2_v2 \
    --dataset sst2_v2 \
    --n_layer 4 \
    --d_model 256 \
    --n_head 8 \
    --d_head 64 \
    --d_inner 512 \
    --dropout 0.1 \
    --dropatt 0.0 \
    --optim adam \
    --lr 1e-4 \
    --warmup_step 0 \
    --max_step 5000 \
    --eval-interval 500 \
    --log-interval 100 \
    --tgt_len 512 \
    --mem_len 128 \
    --eval_tgt_len 128 \
    --batch_size 16 \
    --work_dir smoe_gradually_seed$2 \
    --seed $2 \
    --pretrained_weight $1 \
    --moe --moe-num-expert 16 --moe-top-k 2 \
    --gate_name CustomNaiveGate \
    --dynamic_moe \
    --dynamic_moe_mode linear_increase \
    --dynamic_overall_steps 5000 \
    --moe-top-k-min 16 \
    --moe-top-k-max 16 > log_sst2_seed$2_smoe_gradually.out 2>&1 &