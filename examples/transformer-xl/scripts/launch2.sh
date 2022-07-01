# KEY=layer4_experts16_moe_random
# nohup bash scripts/${KEY}.sh train --work_dir ${KEY} > log_0611_${KEY}.out 2>&1 &




# KEY=layer4_experts16_moe_naive_fix_weight_top_all
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/${KEY}.sh train --work_dir ${KEY} > log_0629_${KEY}.out 2>&1 &
# KEY=layer4_experts16_moe_naive_fix_weight_top2
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/${KEY}.sh train --work_dir ${KEY} > log_0629_${KEY}.out 2>&1 &




KEY=layer4_experts16_moe_dense_fix_weight
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/${KEY}.sh train --work_dir ${KEY} > log_0701_${KEY}.out 2>&1 &


