import os 
import sys
import matplotlib.pyplot as plt 


eval_losses = []
with open(sys.argv[1]) as f:
    data = f.readlines()

for line in data:
    if '| Eval' in line:
        print(line)
        eval_losses.append(float(line[-8:-1]))

print(eval_losses)

