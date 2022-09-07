DATE=0827
GATE_NAME=CustomNaiveGate
Dynamic_mode=linear_increase
NUM_LAYER=4
NUM_EXPERT=16
MIN_K=8
MAX_K=16
SAVE_DIR=test
bash scripts/exp0827/moe_freeze_router.sh \
    ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} 