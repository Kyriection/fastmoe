if [[ $1 == 'exp1' ]]; then
    echo 'Run Experiment1...'
    DATE=0717
    GATE_NAME=CustomNaiveGate
    MoE_INDEX=0,1,3,4,6,7,9,10
    Dynamic_mode=linear_increase
    Dynamic_steps=400000
    SAVE_DIR=Layer12_Experts16_${GATE_NAME}_SFMIX8_${Dynamic_mode}_${Dynamic_steps}
    nohup bash scripts/new/layer12_experts16_moe_freeze_router_multigpu.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp2' ]]; then
    echo 'Run Experiment2...'
    DATE=0717
    GATE_NAME=CustomNaiveGate
    MoE_INDEX=0,1,3,4,6,7,9,10
    Dynamic_mode=linear_decrease
    Dynamic_steps=400000
    SAVE_DIR=Layer12_Experts16_${GATE_NAME}_SFMIX8_${Dynamic_mode}_${Dynamic_steps}
    nohup bash scripts/new/layer12_experts16_moe_freeze_router_multigpu.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp3' ]]; then
    echo 'Run Experiment3...'
    DATE=0717
    GATE_NAME=CustomNaiveGate
    MoE_INDEX=0,1,3,4,6,7,9,10
    Dynamic_mode=cosine_increase
    Dynamic_steps=400000
    SAVE_DIR=Layer12_Experts16_${GATE_NAME}_SFMIX8_${Dynamic_mode}_${Dynamic_steps}
    nohup bash scripts/new/layer12_experts16_moe_freeze_router_multigpu.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp4' ]]; then
    echo 'Run Experiment4...'
    DATE=0717
    GATE_NAME=CustomNaiveGate
    MoE_INDEX=0,1,3,4,6,7,9,10
    Dynamic_mode=cosine_decrease
    Dynamic_steps=400000
    SAVE_DIR=Layer12_Experts16_${GATE_NAME}_SFMIX8_${Dynamic_mode}_${Dynamic_steps}
    nohup bash scripts/new/layer12_experts16_moe_freeze_router_multigpu.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp5' ]]; then
    echo 'Run Experiment5...'
    DATE=0717
    GATE_NAME=CustomNaiveGate
    MoE_INDEX=0,1,3,4,6,7,9,10
    Dynamic_mode=linear_increase
    Dynamic_steps=200000
    SAVE_DIR=Layer12_Experts16_${GATE_NAME}_SFMIX8_${Dynamic_mode}_${Dynamic_steps}
    nohup bash scripts/new/layer12_experts16_moe_freeze_router_multigpu.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp6' ]]; then
    echo 'Run Experiment6...'
    DATE=0717
    GATE_NAME=CustomNaiveGate
    MoE_INDEX=0,1,3,4,6,7,9,10
    Dynamic_mode=linear_decrease
    Dynamic_steps=200000
    SAVE_DIR=Layer12_Experts16_${GATE_NAME}_SFMIX8_${Dynamic_mode}_${Dynamic_steps}
    nohup bash scripts/new/layer12_experts16_moe_freeze_router_multigpu.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp7' ]]; then
    echo 'Run Experiment7...'
    DATE=0717
    GATE_NAME=CustomDropGate
    MoE_INDEX=0,1,3,4,6,7,9,10
    Dynamic_mode=linear_increase
    Dynamic_steps=400000
    SAVE_DIR=Layer12_Experts16_${GATE_NAME}_SFMIX8_${Dynamic_mode}_${Dynamic_steps}
    nohup bash scripts/new/layer12_experts16_moe_freeze_router_multigpu.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp8' ]]; then
    echo 'Run Experiment8...'
    DATE=0717
    GATE_NAME=CustomDropGate
    MoE_INDEX=0,1,3,4,6,7,9,10
    Dynamic_mode=linear_decrease
    Dynamic_steps=400000
    SAVE_DIR=Layer12_Experts16_${GATE_NAME}_SFMIX8_${Dynamic_mode}_${Dynamic_steps}
    nohup bash scripts/new/layer12_experts16_moe_freeze_router_multigpu.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

else
    echo 'unknown argment 1'
fi

