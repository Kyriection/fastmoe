# key=layer4_experts16_moe_dts_random_top1-enwik8
# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/part_moe/layer4_eval.sh CustomDTSRandomGate ${key} 0,1,2,3 > log_EVAL_${key}.out 2>&1 &
# key=layer4_experts16_moe_dts_random_top2-enwik8
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/part_moe/layer4_eval.sh CustomDTSRandomGate ${key} 0,1,2,3 > log_EVAL_${key}.out 2>&1 &
# key=layer4_experts16_moe_naive_middle1-enwik8
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/part_moe/layer4_eval.sh CustomNaiveGate ${key} 1 > log_EVAL_${key}.out 2>&1 &




key=layer4_experts16_moe_naive_freeze_middle2-enwik8
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/part_moe/layer4_eval.sh CustomNaiveGate ${key} 2 > log_EVAL_${key}.out 2>&1 &
key=layer4_experts16_moe_naive_freeze-enwik8
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/part_moe/layer4_eval.sh CustomNaiveGate ${key} 0,1,2,3 > log_EVAL_${key}.out 2>&1 &
key=layer4_experts16_moe_naive_every2-enwik8 
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/part_moe/layer4_eval.sh CustomNaiveGate ${key} 1,3 > log_EVAL_${key}.out 2>&1 &
key=layer4_experts16_moe_naive-enwik8
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/part_moe/layer4_eval.sh CustomNaiveGate ${key} 0,1,2,3 > log_EVAL_${key}.out 2>&1 &
key=layer4_experts16_moe_gradual_naive_reverse-enwik8
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/part_moe/layer4_eval.sh CustomNaiveGate ${key} 0,1,2,3 > log_EVAL_${key}.out 2>&1 &
key=layer4_experts16_moe_gradual_naive-enwik8
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/part_moe/layer4_eval.sh CustomNaiveGate ${key} 0,1,2,3 > log_EVAL_${key}.out 2>&1 &
key=layer4_experts16_moe_dts_top1-enwik8
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/part_moe/layer4_eval.sh CustomDTSGate ${key} 0,1,2,3 > log_EVAL_${key}.out 2>&1 &
key=layer4_experts16_moe_dts_top2-enwik8
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/part_moe/layer4_eval.sh CustomDTSGate ${key} 0,1,2,3 > log_EVAL_${key}.out 2>&1 &
key=layer4_experts16_moe_random-enwik8
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/part_moe/layer4_eval.sh CustomRandomGate ${key} 0,1,2,3 > log_EVAL_${key}.out 2>&1 &
key=layer4_experts16_moe_gradual_random-enwik8
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/part_moe/layer4_eval.sh CustomRandomGate ${key} 0,1,2,3 > log_EVAL_${key}.out 2>&1 &
key=layer4_experts16_moe_gradual_random_reverse-enwik8
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/part_moe/layer4_eval.sh CustomRandomGate ${key} 0,1,2,3 > log_EVAL_${key}.out 2>&1 &
key=layer4_experts16_moe_dropout-enwik8
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/part_moe/layer4_eval.sh CustomDropGate ${key} 0,1,2,3 > log_EVAL_${key}.out 2>&1 &



