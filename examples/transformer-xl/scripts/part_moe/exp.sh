key=layer4_experts16_moe_dts_random_top1-enwik8
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/part_moe/layer4_eval.sh CustomDTSRandomGate ${key} 0,1,2,3 > log_EVAL_${key}.out 2>&1 &

key=layer4_experts16_moe_dts_random_top2-enwik8
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/part_moe/layer4_eval.sh CustomDTSRandomGate ${key} 0,1,2,3 > log_EVAL_${key}.out 2>&1 &

key=layer4_experts16_moe_naive_middle1-enwik8
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/part_moe/layer4_eval.sh CustomNaiveGate ${key} 1 > log_EVAL_${key}.out 2>&1 &






