import os 
import sys
import numpy as np 
import matplotlib.pyplot as plt 


eval_losses = []
with open(sys.argv[1]) as f:
    data = f.readlines()

for line in data:
    if '| Eval' in line:
        print(line)
        eval_losses.append(float(line[-8:-1]))

print(eval_losses)
x = list(np.array(1, len(eval_losses)+1)* 4000)


plt.plot(x, eval_losses)
plt.ylabel('Eval: bpc')
plt.xlabel('Iterations')
plt.title(sys.argv[1])
plt.savefig('{}.png'.format(sys.argv[1][4:-5]), bbox_inches='tight')
plt.close()


