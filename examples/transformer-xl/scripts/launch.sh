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


# moe_index=0
# name=early
# KEY=layer4_experts16_moe_random
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/part_moe/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &
# KEY=layer4_experts16_moe_naive
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/part_moe/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &
# moe_index=3
# name=late
# KEY=layer4_experts16_moe_random
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/part_moe/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &
# KEY=layer4_experts16_moe_naive
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/part_moe/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &


# moe_index=1
# name=middle1
# KEY=layer4_experts16_moe_random
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/part_moe/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &
# KEY=layer4_experts16_moe_naive
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/part_moe/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &
# moe_index=2
# name=middle2
# KEY=layer4_experts16_moe_random
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/part_moe/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &
# KEY=layer4_experts16_moe_naive
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/part_moe/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &


# if [[ $1 == 'exp1' ]]; then
#     echo 'Run Experiment1...'
#     moe_index=2
#     name=middle2
#     KEY=layer4_experts16_moe_random
#     nohup bash scripts/part_moe_gpus/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &

# elif [[ $1 == 'exp2' ]]; then
#     echo 'Run Experiment2...'
#     moe_index=2
#     name=middle2
#     KEY=layer4_experts16_moe_naive
#     nohup bash scripts/part_moe_gpus/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &

# elif [[ $1 == 'exp3' ]]; then
#     echo 'Run Experiment3...'
#     moe_index=1,3
#     name=every2
#     KEY=layer4_experts16_moe_naive_freeze
#     nohup bash scripts/part_moe_gpus/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &

# elif [[ $1 == 'exp4' ]]; then
#     echo 'Run Experiment4...'
#     moe_index=0
#     name=early
#     KEY=layer4_experts16_moe_naive_freeze
#     nohup bash scripts/part_moe_gpus/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &

# elif [[ $1 == 'exp5' ]]; then
#     echo 'Run Experiment5...'
#     moe_index=3
#     name=late
#     KEY=layer4_experts16_moe_naive_freeze
#     nohup bash scripts/part_moe_gpus/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &

# elif [[ $1 == 'exp6' ]]; then
#     echo 'Run Experiment6...'
#     moe_index=1
#     name=middle1
#     KEY=layer4_experts16_moe_naive_freeze
#     nohup bash scripts/part_moe_gpus/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &

# elif [[ $1 == 'exp7' ]]; then
#     echo 'Run Experiment7...'
#     moe_index=2
#     name=middle2
#     KEY=layer4_experts16_moe_naive_freeze
#     nohup bash scripts/part_moe_gpus/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0621_${KEY}_${name}.out 2>&1 &

# else
#     echo 'unknown argment 1'
# fi



if [[ $1 == 'exp1' ]]; then
    echo 'Run Experiment1...'
    moe_index=1,3,5,6,7,11
    name=every2
    KEY=layer12_experts16_moe_naive
    nohup bash scripts/part_moe_gpus/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0702_${KEY}_${name}.out 2>&1 &

elif [[ $1 == 'exp2' ]]; then
    echo 'Run Experiment2...'
    moe_index=0
    name=early
    KEY=layer12_experts16_moe_naive
    nohup bash scripts/part_moe_gpus/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0702_${KEY}_${name}.out 2>&1 &

elif [[ $1 == 'exp3' ]]; then
    echo 'Run Experiment3...'
    moe_index=6
    name=middle
    KEY=layer12_experts16_moe_naive
    nohup bash scripts/part_moe_gpus/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0702_${KEY}_${name}.out 2>&1 &

elif [[ $1 == 'exp4' ]]; then
    echo 'Run Experiment4...'
    moe_index=11
    name=late
    KEY=layer12_experts16_moe_naive
    nohup bash scripts/part_moe_gpus/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0702_${KEY}_${name}.out 2>&1 &

elif [[ $1 == 'exp5' ]]; then
    echo 'Run Experiment5...'
    moe_index=0,1,3,4,6,7,9,10
    name=s2f1_x4
    KEY=layer12_experts16_moe_naive
    nohup bash scripts/part_moe_gpus/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0702_${KEY}_${name}.out 2>&1 &

elif [[ $1 == 'exp6' ]]; then
    echo 'Run Experiment6...'
    moe_index=2,5,8,11
    name=f2s1_x4
    KEY=layer12_experts16_moe_naive
    nohup bash scripts/part_moe_gpus/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0702_${KEY}_${name}.out 2>&1 &

elif [[ $1 == 'exp7' ]]; then
    echo 'Run Experiment7...'
    moe_index=5,11
    name=f5s1_x2
    KEY=layer12_experts16_moe_naive
    nohup bash scripts/part_moe_gpus/${KEY}.sh train ${moe_index} --work_dir ${KEY}_${name} > log_0702_${KEY}_${name}.out 2>&1 &

elif [[ $1 == 'exp8' ]]; then
    echo 'Run Experiment8...'
    KEY=layer12_experts16_big_dense_dropout
    nohup bash scripts/part_moe_gpus/${KEY}.sh train --work_dir ${KEY} > log_0703_${KEY}.out 2>&1 &

elif [[ $1 == 'exp9' ]]; then
    echo 'Run Experiment9...'
    KEY=layer4_experts16_big_dense_dropout
    nohup bash scripts/part_moe/${KEY}.sh train --work_dir ${KEY} > log_0703_${KEY}.out 2>&1 &

elif [[ $1 == 'exp10' ]]; then
    echo 'Run Experiment10...'
    KEY=layer4_experts16_moe_dense_fix_allweight
    nohup bash scripts/part_moe/${KEY}.sh train --work_dir ${KEY} > log_0703_${KEY}.out 2>&1 &

elif [[ $1 == 'exp11' ]]; then
    echo 'Run Experiment11...'
    KEY=layer4_experts16_moe_gradual_naive_reverse
    nohup bash scripts/part_moe/${KEY}.sh train --work_dir ${KEY} > log_0706_${KEY}.out 2>&1 &

elif [[ $1 == 'exp12' ]]; then
    echo 'Run Experiment12...'
    KEY=layer4_experts16_moe_gradual_naive
    nohup bash scripts/part_moe/${KEY}.sh train --work_dir ${KEY} > log_0706_${KEY}.out 2>&1 &

elif [[ $1 == 'exp13' ]]; then
    echo 'Run Experiment13...'
    KEY=layer4_experts16_moe_gradual_random_reverse
    nohup bash scripts/part_moe/${KEY}.sh train --work_dir ${KEY} > log_0706_${KEY}.out 2>&1 &

else
    echo 'unknown argment 1'
fi

