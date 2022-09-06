CUDA_VISIBLE_DEVICES=0 nohup bash scripts/exp0827/fine_tuning_sst2.sh $1 1e-4 $2 > log_dense_$2_lr1e-4_sst2.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/exp0827/fine_tuning_sst2.sh $1 1e-5 $2 > log_dense_$2_lr1e-5_sst2.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/exp0827/fine_tuning_sst2.sh $1 5e-5 $2 > log_dense_$2_lr5e-5_sst2.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/exp0827/fine_tuning_sst2.sh $1 5e-4 $2 > log_dense_$2_lr5e-4_sst2.out 2>&1 &