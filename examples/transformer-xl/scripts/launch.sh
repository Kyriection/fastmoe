CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_enwik8_base_naivemoe.sh train naivemoe > log_naivemoe.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_enwik8_base_naivemoe_freeze.sh train naivemoe_freeze > log_naivemoe_freeze.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/run_enwik8_base_randassignmoe.sh train randassignmoe > log_randassignmoe.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/run_enwik8_base_dropmoe.sh train dropmoe > log_dropmoe.out 2>&1 &