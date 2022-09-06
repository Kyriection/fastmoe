# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/exp0827/finetuning_dense.sh $1 1e-4 $2 > log_dense_$2_lr1e-4_sst2.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/exp0827/finetuning_dense.sh $1 1e-5 $2 > log_dense_$2_lr1e-5_sst2.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/exp0827/finetuning_dense.sh $1 5e-5 $2 > log_dense_$2_lr5e-5_sst2.out 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/exp0827/finetuning_dense.sh $1 5e-4 $2 > log_dense_$2_lr5e-4_sst2.out 2>&1 &



# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/exp0827/finetuning_dense.sh $1 2e-4 $2 > log_dense_$2_lr2e-4_sst2.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/exp0827/finetuning_dense.sh $1 3e-4 $2 > log_dense_$2_lr3e-4_sst2.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/exp0827/finetuning_dense.sh $1 7e-5 $2 > log_dense_$2_lr7e-5_sst2.out 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/exp0827/finetuning_dense.sh $1 9e-5 $2 > log_dense_$2_lr9e-5_sst2.out 2>&1 &



# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/exp0827/finetune_ablation.sh $1 1e-4 $2 256 > log_dense_$2_sst2_tgtlen256.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/exp0827/finetune_ablation.sh $1 1e-4 $2 128 > log_dense_$2_sst2_tgtlen128.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/exp0827/finetune_ablation.sh $1 1e-4 $2 64 > log_dense_$2_sst2_tgtlen64.out 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/exp0827/finetune_ablation.sh $1 1e-4 $2 32 > log_dense_$2_sst2_tgtlen32.out 2>&1 &



# CUDA_VISIBLE_DEVICES=$3 nohup bash scripts/exp0827/finetuning_dense.sh $1 1e-4 $2 > log_dense_$2_lr1e-4_sst2.out 2>&1 &



# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/exp0827/fine_tuning_sst2_naive.sh $1 naive_moe_dense_finetuning 16 > log_naive_moe_dense_finetuning_sst2.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/exp0827/fine_tuning_sst2_naive.sh $1 naive_moe_gradual_finetuning 8 > log_naive_moe_gradual_finetuning_8_sst2.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/exp0827/fine_tuning_sst2_naive.sh random naive_moe_dense_finetuning_randominit 16 > log_naive_moe_dense_finetuning_sst2_randominit.out 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/exp0827/fine_tuning_sst2_naive.sh random naive_moe_gradual_finetuning_randominit 8 > log_naive_moe_gradual_finetuning_8_sst2_randominit.out 2>&1 &




CUDA_VISIBLE_DEVICES=0 nohup bash scripts/exp0827/fine_tuning_sst2_freeze.sh $1 RandWeight_moe_dense_finetuning 16 > log_RandWeight_moe_dense_finetuning_sst2.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/exp0827/fine_tuning_sst2_freeze.sh $1 RandWeight_moe_gradual_finetuning 8 > log_RandWeight_moe_gradual_finetuning_8_sst2.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/exp0827/fine_tuning_sst2_freeze.sh random RandWeight_moe_dense_finetuning_randominit 16 > log_RandWeight_moe_dense_finetuning_sst2_randominit.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/exp0827/fine_tuning_sst2_freeze.sh random RandWeight_moe_gradual_finetuning_randominit 8 > log_RandWeight_moe_gradual_finetuning_8_sst2_randominit.out 2>&1 &
