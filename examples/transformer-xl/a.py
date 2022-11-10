import numpy as np 

min_experts = 8
max_experts = 16
overall_steps = 100000
t = []
for steps in range(overall_steps):

    number_experts = min_experts - max_experts
    current_steps = steps // (overall_steps // 300)
    cosine_value = 0.99 ** current_steps
    gate_num = round(number_experts * cosine_value) + max_experts
    t.append(gate_num)


import matplotlib.pyplot as plt 


plt.plot(t)
plt.show()