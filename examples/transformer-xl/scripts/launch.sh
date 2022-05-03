# CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_enwik8_base_naivemoe.sh train --work_dir naivemoe > log_naivemoe.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_enwik8_base_naivemoe_freeze.sh train --work_dir naivemoe_freeze > log_naivemoe_freeze.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/run_enwik8_base_randassignmoe.sh train --work_dir randassignmoe > log_randassignmoe.out 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/run_enwik8_base_dropmoe.sh train --work_dir dropmoe > log_dropmoe.out 2>&1 &



# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/run_enwik8_baseline.sh train --work_dir baseline > log_baseline.out 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup bash scripts/run_enwik8_basic.sh train --work_dir basic > log_basic.out 2>&1 &






CUDA_VISIBLE_DEVICES=$1 bash scripts/run_enwik8_base_randassignmoe_new.sh train --work_dir randassignmoe_new










