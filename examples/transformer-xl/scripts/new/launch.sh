# #####################  UT VITA1 #################
# DATE=0714
# GATE_NAME=CustomNaiveGate
# MoE_INDEX=0,1,2,3
# Dynamic_mode=linear_increase
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_${Dynamic_mode}_${Dynamic_steps}
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/new/layer4_experts16_moe_freeze_router.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0714
# GATE_NAME=CustomNaiveGate
# MoE_INDEX=0,1,2,3
# Dynamic_mode=linear_decrease
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_${Dynamic_mode}_${Dynamic_steps}
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/new/layer4_experts16_moe_freeze_router.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0714
# GATE_NAME=CustomNaiveGate
# MoE_INDEX=0,1,2,3
# Dynamic_mode=cosine_decrease
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_${Dynamic_mode}_${Dynamic_steps}
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/new/layer4_experts16_moe_freeze_router.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0714
# GATE_NAME=CustomNaiveGate
# MoE_INDEX=0,1,2,3
# Dynamic_mode=cosine_increase
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_${Dynamic_mode}_${Dynamic_steps}
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/new/layer4_experts16_moe_freeze_router.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &








# #####################  UT VITA3 #################
# DATE=0714
# GATE_NAME=CustomNaiveGate
# MoE_INDEX=0,1,2,3
# Dynamic_mode=linear_increase
# Dynamic_steps=200000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_${Dynamic_mode}_${Dynamic_steps}
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/new/layer4_experts16_moe_freeze_router.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0714
# GATE_NAME=CustomNaiveGate
# MoE_INDEX=0,1,2,3
# Dynamic_mode=linear_decrease
# Dynamic_steps=200000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_${Dynamic_mode}_${Dynamic_steps}
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/new/layer4_experts16_moe_freeze_router.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0714
# GATE_NAME=CustomNaiveGate
# MoE_INDEX=1,2,3
# Dynamic_mode=linear_increase
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_123_${Dynamic_mode}_${Dynamic_steps}
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/new/layer4_experts16_moe_freeze_router.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0714
# GATE_NAME=CustomNaiveGate
# MoE_INDEX=1,2,3
# Dynamic_mode=linear_decrease
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_123_${Dynamic_mode}_${Dynamic_steps}
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/new/layer4_experts16_moe_freeze_router.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &








# #####################  UT VITA4 #################
# DATE=0714
# GATE_NAME=CustomNaiveGate
# MoE_INDEX=1,3
# Dynamic_mode=linear_increase
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_13_${Dynamic_mode}_${Dynamic_steps}
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/new/layer4_experts16_moe_freeze_router.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0714
# GATE_NAME=CustomNaiveGate
# MoE_INDEX=1,3
# Dynamic_mode=linear_decrease
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_13_${Dynamic_mode}_${Dynamic_steps}
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/new/layer4_experts16_moe_freeze_router.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0714
# GATE_NAME=CustomDropGate
# MoE_INDEX=0,1,2,3
# Dynamic_mode=linear_increase
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_${Dynamic_mode}_${Dynamic_steps}
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/new/layer4_experts16_moe_freeze_router.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0714
# GATE_NAME=CustomDropGate
# MoE_INDEX=0,1,2,3
# Dynamic_mode=linear_decrease
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_${Dynamic_mode}_${Dynamic_steps}
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/new/layer4_experts16_moe_freeze_router.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &





#####################  Tamu VITA5 #################
DATE=0714
DIM=6400
SAVE_DIR=Layer4_Experts16_Big_Dense_${DIM}
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash scripts/new/layer4_exprts16_dense_multigpu.sh ${DIM} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

DATE=0714
DIM=5594
SAVE_DIR=Layer4_Experts16_Big_Dense_${DIM}
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash scripts/new/layer4_exprts16_dense_multigpu.sh ${DIM} > log_${DATE}_${SAVE_DIR}.out 2>&1 &



# DATE=0714
# DIM=4608
# SAVE_DIR=Layer4_Experts16_Big_Dense_${DIM}
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/new/layer4_experts16_dense.sh ${DIM} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0714
# DIM=2816
# SAVE_DIR=Layer4_Experts16_Big_Dense_${DIM}
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/new/layer4_experts16_dense.sh ${DIM} > log_${DATE}_${SAVE_DIR}.out 2>&1 &




