KEY=layer4_experts16_moe_random
nohup bash scripts/${KEY}.sh train --work_dir ${KEY} > log_0611_${KEY}.out 2>&1 &
