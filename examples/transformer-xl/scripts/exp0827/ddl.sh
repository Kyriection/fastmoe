# echo 'Run SMOE TOP-12'
# DATE=0919
# GATE_NAME=CustomNaiveGate
# Dynamic_mode=linear_increase
# NUM_LAYER=4
# NUM_EXPERT=16
# MIN_K=12
# MAX_K=12
# SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
# nohup bash scripts/exp0827/moe.sh \
#     ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &


# echo 'Run RMT 8-12'
# DATE=0919
# GATE_NAME=CustomNaiveGate
# Dynamic_mode=linear_increase
# NUM_LAYER=8
# NUM_EXPERT=16
# MIN_K=4
# MAX_K=16
# SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_RandomWeight_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash scripts/exp0827/moe_freeze_router.sh \
#     ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &



echo 'Run RMT TOP-12'
DATE=0921
GATE_NAME=CustomNaiveGate
Dynamic_mode=linear_increase
NUM_LAYER=4
NUM_EXPERT=16
MIN_K=12
MAX_K=12
SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_RandomWeight_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
nohup bash scripts/exp0827/moe_freeze_router.sh \
    ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &


# echo 'Run SMOE TOP-4'
# DATE=0919
# GATE_NAME=CustomNaiveGate
# Dynamic_mode=linear_increase
# NUM_LAYER=4
# NUM_EXPERT=16
# MIN_K=4
# MAX_K=4
# SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
# nohup bash scripts/exp0827/moe.sh \
#     ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &



# echo 'Run SMOE TOP-6'
# DATE=0919
# GATE_NAME=CustomNaiveGate
# Dynamic_mode=linear_increase
# NUM_LAYER=4
# NUM_EXPERT=16
# MIN_K=6
# MAX_K=6
# SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
# nohup bash scripts/exp0827/moe.sh \
#     ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &


# echo 'Run SMOE TOP-8'
# DATE=0919
# GATE_NAME=CustomNaiveGate
# Dynamic_mode=linear_increase
# NUM_LAYER=4
# NUM_EXPERT=16
# MIN_K=8
# MAX_K=8
# SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
# nohup bash scripts/exp0827/moe.sh \
#     ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &


# echo 'Run SMOE TOP-10'
# DATE=0919_Test
# GATE_NAME=CustomNaiveGate
# Dynamic_mode=linear_increase
# NUM_LAYER=4
# NUM_EXPERT=16
# MIN_K=10
# MAX_K=10
# SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
# nohup bash scripts/exp0827/moe.sh \
#     ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &
