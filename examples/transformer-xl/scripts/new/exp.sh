# ## VITA4 UT ##
# DATE=0720
# GATE_NAME=CustomNaiveGate
# MoE_INDEX=0,1,2,3
# Dynamic_mode=cosine_decrease
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_${Dynamic_mode}_${Dynamic_steps}
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/new/layer4_experts16_moe_freeze_router.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0720
# GATE_NAME=CustomDropGate
# MoE_INDEX=0,1,2,3
# Dynamic_mode=cosine_decrease
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_${Dynamic_mode}_${Dynamic_steps}
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/new/layer4_experts16_moe_freeze_router.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0720
# GATE_NAME=CustomNaiveGate
# MoE_INDEX=0,1,2,3
# Dynamic_mode=cosine_decrease
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_${Dynamic_mode}_${Dynamic_steps}_STABLE
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/new/layer4_experts16_moe_router_stable.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0720
# GATE_NAME=CustomNaiveGate
# MoE_INDEX=0,1,2,3
# Dynamic_mode=cosine_decrease
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_${Dynamic_mode}_${Dynamic_steps}_SWAD
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/new/layer4_experts16_moe_freeze_router_swa.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &


# ## VITA3 UT ##
# DATE=0720
# GATE_NAME=CustomRandomGate
# MoE_INDEX=0,1,2,3
# Dynamic_mode=cosine_decrease
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_${Dynamic_mode}_${Dynamic_steps}
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/new/layer4_experts16_moe.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0720
# GATE_NAME=CustomRandomGate
# MoE_INDEX=0,1,2,3
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/new/layer4_experts16_moe_basic.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &


# DATE=0720
# GATE_NAME=CustomRandomGate
# MoE_INDEX=0,1,2,3
# Dynamic_mode=cosine_decrease
# Dynamic_steps=400000
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_${Dynamic_mode}_${Dynamic_steps}_SWAD
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/new/layer4_experts16_moe_swa.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${Dynamic_mode} ${Dynamic_steps} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0720
# GATE_NAME=CustomRandomGate
# MoE_INDEX=0,1,2,3
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_SWAD
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/new/layer4_experts16_moe_basic_swa.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &




# ## VITA1 UT ##
# DATE=0720
# GATE_NAME=CustomNaiveGate
# MoE_INDEX=0,1,2,3
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_SWA
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/new/layer4_experts16_moe_basic_stable.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

# DATE=0720
# GATE_NAME=CustomNaiveGate
# MoE_INDEX=0,1,2,3
# SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_STABLE
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/new/layer4_experts16_moe_basic_swa.sh \
#     ${GATE_NAME} ${MoE_INDEX} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &


## VITA2 UT ##
DATE=0720
GATE_NAME=CustomNaiveGate
MoE_INDEX=0,1,2,3
SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_FREEZE
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/new/layer4_experts16_moe_freeze_basic.sh \
    ${GATE_NAME} ${MoE_INDEX} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

DATE=0720
GATE_NAME=CustomNaiveGate
MoE_INDEX=0,1,2,3
SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_Basic
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/new/layer4_experts16_moe_basic.sh \
    ${GATE_NAME} ${MoE_INDEX} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

DATE=0720
GATE_NAME=CustomDropGate
MoE_INDEX=0,1,2,3
SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_FREEZE
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/new/layer4_experts16_moe_freeze_basic.sh \
    ${GATE_NAME} ${MoE_INDEX} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &

DATE=0720
GATE_NAME=CustomDropGate
MoE_INDEX=0,1,2,3
SAVE_DIR=Layer4_Experts16_${GATE_NAME}_ALL_Basic
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/new/layer4_experts16_moe_basic.sh \
    ${GATE_NAME} ${MoE_INDEX} ${SAVE_DIR} > log_${DATE}_${SAVE_DIR}.out 2>&1 &
