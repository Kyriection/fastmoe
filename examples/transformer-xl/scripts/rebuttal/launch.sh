CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/rebuttal/test_ex2.sh > log_rebuttal_test_Ex2.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u scripts/rebuttal/test_ex4.sh > log_rebuttal_test_Ex4.out &
CUDA_VISIBLE_DEVICES=2 nohup python -u scripts/rebuttal/test_ex8.sh > log_rebuttal_test_Ex8.out &
CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/rebuttal/test_ex16.sh > log_rebuttal_test_Ex16.out &