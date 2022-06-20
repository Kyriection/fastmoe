# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_enwik8_base_naivemoe.sh train --work_dir naivemoe > log_naivemoe.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_enwik8_base_naivemoe_freeze.sh train --work_dir naivemoe_freeze > log_naivemoe_freeze.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/run_enwik8_base_randassignmoe.sh train --work_dir randassignmoe > log_randassignmoe.out 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/run_enwik8_base_dropmoe.sh train --work_dir dropmoe > log_dropmoe.out 2>&1 &



# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/run_enwik8_baseline.sh train --work_dir baseline > log_baseline.out 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/run_enwik8_basic.sh train --work_dir basic > log_basic.out 2>&1 &






# CUDA_VISIBLE_DEVICES=$1 nohup bash scripts/run_enwik8_base_randassignmoe_new.sh train --work_dir randassignmoe_new > log_randassignmoe_new.out 2>&1 &


# KEY=layer4_experts16_big_dense
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/${KEY}.sh train --work_dir ${KEY} > log_0620_${KEY}.out 2>&1 &
# KEY=layer4_experts16_dense
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/${KEY}.sh train --work_dir ${KEY} > log_0620_${KEY}.out 2>&1 &
# KEY=layer4_experts16_moe_dropout
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/${KEY}.sh train --work_dir ${KEY} > log_0611_${KEY}.out 2>&1 &



# KEY=layer4_experts16_moe_gradual_random
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/${KEY}.sh train --work_dir ${KEY} > log_0611_${KEY}.out 2>&1 &
# KEY=layer4_experts16_moe_naive_freeze
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/${KEY}.sh train --work_dir ${KEY} > log_0611_${KEY}.out 2>&1 &
# KEY=layer4_experts16_moe_naive
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/${KEY}.sh train --work_dir ${KEY} > log_0611_${KEY}.out 2>&1 &
# KEY=layer4_experts16_moe_random
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/${KEY}.sh train --work_dir ${KEY} > log_0611_${KEY}.out 2>&1 &



# KEY=layer4_experts16_moe_naive
# CUDA_VISIBLE_DEVICES=3 bash scripts/${KEY}.sh train --work_dir ${KEY}


# KEY=layer4_experts16_moe_dropout
# nohup bash scripts/${KEY}.sh train --work_dir ${KEY} > log_0611_${KEY}.out 2>&1 &





# moe_index=1,3
# name=every2
# KEY=layer4_experts16_moe_random
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/part_moe/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &
# KEY=layer4_experts16_moe_naive
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/part_moe/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &


moe_index=0
name=early
KEY=layer4_experts16_moe_random
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/part_moe/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &
KEY=layer4_experts16_moe_naive
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/part_moe/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &
moe_index=3
name=late
KEY=layer4_experts16_moe_random
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/part_moe/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &
KEY=layer4_experts16_moe_naive
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/part_moe/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &








