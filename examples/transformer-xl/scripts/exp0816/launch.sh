#!/bin/bash
if [[ $1 == 'exp1' ]]; then
    echo 'Run Experiment 1'
    DATE=0816
    GATE_NAME=CustomNaiveGate
    MoE_INDEX=0,1,2,3
    Dynamic_mode=linear_increase
    MIN_K=2
    MAX_K=16
    SAVE_DIR=MoE4-16_RandomWeight_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0816/layer4_experts16_moe_freeze_router.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &
elif [[ $1 == 'exp2' ]]; then
    echo 'Run Experiment 2'
    DATE=0816
    GATE_NAME=CustomNaiveGate
    MoE_INDEX=0,1,2,3
    Dynamic_mode=linear_decrease
    MIN_K=16
    MAX_K=16
    SAVE_DIR=MoE4-16_RandomWeight_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0816/layer4_experts16_moe_freeze_router.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &
elif [[ $1 == 'exp3' ]]; then
    echo 'Run Experiment 3'
    DATE=0816
    GATE_NAME=CustomNaiveGate
    MoE_INDEX=0,1,2,3
    Dynamic_mode=linear_decrease
    MIN_K=12
    MAX_K=16
    SAVE_DIR=MoE4-16_RandomWeight_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0816/layer4_experts16_moe_freeze_router.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &
elif [[ $1 == 'exp4' ]]; then
    echo 'Run Experiment 4'
    DATE=0816
    GATE_NAME=CustomNaiveGate
    MoE_INDEX=0,1,2,3
    Dynamic_mode=linear_decrease
    MIN_K=8
    MAX_K=16
    SAVE_DIR=MoE4-16_RandomWeight_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0816/layer4_experts16_moe_freeze_router.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &
elif [[ $1 == 'exp5' ]]; then
    echo 'Run Experiment 5'
    DATE=0816
    GATE_NAME=CustomNaiveGate
    MoE_INDEX=0,1,2,3
    Dynamic_mode=linear_decrease
    MIN_K=4
    MAX_K=16
    SAVE_DIR=MoE4-16_RandomWeight_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0816/layer4_experts16_moe_freeze_router.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &
elif [[ $1 == 'exp6' ]]; then
    echo 'Run Experiment 6'
    DATE=0816
    GATE_NAME=CustomNaiveGate
    MoE_INDEX=0,1,2,3
    Dynamic_mode=linear_decrease
    MIN_K=2
    MAX_K=16
    SAVE_DIR=MoE4-16_RandomWeight_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0816/layer4_experts16_moe_freeze_router.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &
elif [[ $1 == 'exp7' ]]; then
    echo 'Run Experiment 7'
    DATE=0816
    GATE_NAME=CustomNaiveGate
    MoE_INDEX=0,1,2,3
    Dynamic_mode=linear_increase
    MIN_K=4
    MAX_K=16
    SAVE_DIR=MoE4-16_RandomWeight_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0816/layer4_experts16_moe_freeze_router.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &
elif [[ $1 == 'exp8' ]]; then
    echo 'Run Experiment 8'
    DATE=0816
    GATE_NAME=CustomNaiveGate
    MoE_INDEX=0,1,2,3
    Dynamic_mode=linear_increase
    MIN_K=8
    MAX_K=16
    SAVE_DIR=MoE4-16_RandomWeight_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0816/layer4_experts16_moe_freeze_router.sh \
        ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &
else
    echo 'unknown argment 1'
fi
