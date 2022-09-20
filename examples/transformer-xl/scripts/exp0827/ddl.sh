echo 'Run SMOE TOP-12'
DATE=0919
GATE_NAME=CustomNaiveGate
Dynamic_mode=linear_increase
NUM_LAYER=4
NUM_EXPERT=16
MIN_K=12
MAX_K=12
SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash scripts/exp0827/moe.sh \
    ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &
