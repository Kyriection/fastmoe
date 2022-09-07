if [[ $1 == 'exp1' ]]; then
    echo 'Run Experiment 1'
    DATE=0907
    NUM_LAYER=12
    NUM_DIM=8192
    SAVE_DIR=MoE${NUM_LAYER}-${NUM_DIM}_Dense_${DATE}
    nohup bash scripts/exp0827/dense_single.sh \
        ${NUM_LAYER} ${NUM_DIM} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp2' ]]; then
    echo 'Run Experiment 2'
    DATE=0907
    NUM_LAYER=8
    NUM_DIM=8192
    SAVE_DIR=MoE${NUM_LAYER}-${NUM_DIM}_Dense_${DATE}
    nohup bash scripts/exp0827/dense_single.sh \
        ${NUM_LAYER} ${NUM_DIM} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp3' ]]; then
    echo 'Run Experiment 3'
    DATE=0907
    NUM_LAYER=4
    NUM_DIM=16384
    SAVE_DIR=MoE${NUM_LAYER}-${NUM_DIM}_Dense_${DATE}
    nohup bash scripts/exp0827/dense_single.sh \
        ${NUM_LAYER} ${NUM_DIM} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp4' ]]; then
    echo 'Run Experiment 4'
    DATE=0907
    GATE_NAME=CustomNaiveGate
    Dynamic_mode=linear_increase
    NUM_LAYER=2
    NUM_EXPERT=16
    MIN_K=8
    MAX_K=16
    SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0827/moe.sh \
        ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp5' ]]; then
    echo 'Run Experiment 5'
    DATE=0907
    GATE_NAME=CustomNaiveGate
    Dynamic_mode=linear_increase
    NUM_LAYER=8
    NUM_EXPERT=16
    MIN_K=8
    MAX_K=16
    SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0827/moe.sh \
        ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp6' ]]; then
    echo 'Run Experiment 6'
    DATE=0907
    GATE_NAME=CustomNaiveGate
    Dynamic_mode=linear_increase
    NUM_LAYER=12
    NUM_EXPERT=16
    MIN_K=8
    MAX_K=16
    SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0827/moe.sh \
        ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp7' ]]; then
    echo 'Run Experiment 7'
    DATE=0907
    GATE_NAME=CustomNaiveGate
    Dynamic_mode=linear_increase
    NUM_LAYER=4
    NUM_EXPERT=2
    MIN_K=1
    MAX_K=2
    SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0827/moe.sh \
        ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp8' ]]; then
    echo 'Run Experiment 8'
    DATE=0907
    GATE_NAME=CustomNaiveGate
    Dynamic_mode=linear_increase
    NUM_LAYER=4
    NUM_EXPERT=4
    MIN_K=2
    MAX_K=4
    SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0827/moe.sh \
        ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp9' ]]; then
    echo 'Run Experiment 9'
    DATE=0907
    GATE_NAME=CustomNaiveGate
    Dynamic_mode=linear_increase
    NUM_LAYER=4
    NUM_EXPERT=8
    MIN_K=4
    MAX_K=8
    SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0827/moe.sh \
        ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp10' ]]; then
    echo 'Run Experiment 10'
    DATE=0907
    GATE_NAME=CustomNaiveGate
    Dynamic_mode=linear_increase
    NUM_LAYER=4
    NUM_EXPERT=32
    MIN_K=16
    MAX_K=32
    SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0827/moe.sh \
        ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp11' ]]; then
    echo 'Run Experiment 11'
    DATE=0907
    GATE_NAME=CustomNaiveGate
    Dynamic_mode=linear_increase
    NUM_LAYER=4
    NUM_EXPERT=64
    MIN_K=32
    MAX_K=64
    SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0827/moe.sh \
        ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp12' ]]; then
    echo 'Run Experiment 12'
    DATE=0907
    GATE_NAME=CustomNaiveGate
    Dynamic_mode=linear_increase
    NUM_LAYER=8
    NUM_EXPERT=16
    MIN_K=10
    MAX_K=16
    SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0827/moe_freeze_router.sh \
        ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp13' ]]; then
    echo 'Run Experiment 13'
    DATE=0907
    GATE_NAME=CustomNaiveGate
    Dynamic_mode=linear_increase
    NUM_LAYER=4
    NUM_EXPERT=32
    MIN_K=24
    MAX_K=32
    SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0827/moe_freeze_router.sh \
        ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

elif [[ $1 == 'exp14' ]]; then
    echo 'Run Experiment 14'
    DATE=0907
    GATE_NAME=CustomNaiveGate
    Dynamic_mode=linear_increase
    NUM_LAYER=4
    NUM_EXPERT=64
    MIN_K=48
    MAX_K=64
    SAVE_DIR=MoE${NUM_LAYER}-${NUM_EXPERT}_Naive_ALL_${Dynamic_mode}_${MIN_K}_${MAX_K}
    nohup bash scripts/exp0827/moe_freeze_router.sh \
        ${GATE_NAME} ${Dynamic_mode} ${MIN_K} ${MAX_K} ${SAVE_DIR} ${NUM_LAYER} ${NUM_EXPERT} > log_${DATE}_${SAVE_DIR}.out 2>&1 &
