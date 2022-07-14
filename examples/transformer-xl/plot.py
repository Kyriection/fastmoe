import numpy as np 
import matplotlib.pyplot as plt 

moe = [1.47943,1.45892,1.45977,1.46506,1.47008]
moe_gradual = [1.53697,1.51226,1.51150,1.51535,1.52038]
moe_gradual_increase = [1.85910,1.70752,1.66507,1.65309,1.65021]
moe_every = [1.53393,1.51840,1.51876,1.52110,1.52314]
moe_early = [1.79549,1.79212,1.79246,1.79393,1.79566]
moe_middle1 = [1.79549,1.79212,1.79246,1.79393,1.79566]
moe_late = [1.28949,1.26918,1.27003,1.27382,1.27734]

rand = [1.34854,1.33078,1.32343,1.32024,1.31859]
rand_gradual = [3.18041,1.66447,1.33045,1.25391,1.22845]
rand_gradual_increase = [3.91116,1.84227,1.36504,1.27271,1.24604]
rand_early = [1.32829,1.32734,1.32689,1.32672,1.32660]
rand_middle1 = [1.33121,1.32816,1.32662,1.32593,1.32562]
rand_late = [1.32744,1.32277,1.32040,1.31929,1.31871]

freeze = [1.26292,1.20228,1.22960,1.39330,1.78782]
freeze_every = [1.22855,1.20015,1.20693,1.24402,1.31184]
freeze_early = [1.44975,1.44410,1.44745,1.45832,1.47881]
freeze_middle1 = [1.27156,1.25849,1.26137,1.27442,1.29651]
freeze_middle2 = [1.27869,1.25224,1.26474,1.30401,1.36068]
freeze_late = [1.28920,1.26452,1.26732,1.28394,1.31037]

rand_dts_1 = [1.39351,1.32667,1.30394,1.29475,1.29094] 
rand_dts_2 = [1.39490,1.32858,1.30603,1.29732,1.29353] 
moe_dts_1 = [1.17791,1.17653,1.17647,1.17647,1.17647] 
moe_dts_2 = [1.17626,1.17487,1.17481,1.17481,1.17481] 

x = [1,2,4,8,16]

dense_eval_top2 = 1.28901
dense_eval_all = 1.16041


# plt.grid(linestyle='dashed', zorder=0)
# plt.rcParams['font.sans-serif'] = 'Times New Roman'
# plt.plot(x, moe, label='Learnable_MoE', color='blue')
# plt.plot(x, rand, label='Random_MoE', color='darkorange')
# plt.plot(x, freeze, label='Learnable_MoE-Freeze', color='forestgreen')
# plt.hlines(dense_eval_top2, 1, 16, label='Dense', color='black')
# plt.hlines(dense_eval_all, 1, 16, label='Big-Dense', color='black', linestyle='dashdot')
# plt.xticks(x,x)
# plt.xlabel('Number of Experts')
# plt.ylabel('bpc')
# plt.title('4 Layers, 16 Experts')
# plt.legend()
# plt.savefig('layer4_expert16.png', bbox_inches='tight')
# plt.close()


# plt.grid(linestyle='dashed', zorder=0)
# plt.rcParams['font.sans-serif'] = 'Times New Roman'
# plt.plot(x, moe, label='Learnable_MoE')
# plt.plot(x, moe_gradual, label='Learnable_MoE (Gradually)')
# plt.plot(x, moe_gradual_increase, label='Learnable_MoE (Gradually Decrease)')
# plt.plot(x, moe_every, label='Learnable_MoE (1,3)')
# plt.plot(x, moe_early, label='Learnable_MoE (0)')
# plt.plot(x, moe_middle1, label='Learnable_MoE (1)')
# plt.plot(x, moe_late, label='Learnable_MoE (3)')
# plt.hlines(dense_eval_top2, 1, 16, label='Dense', color='black')
# plt.hlines(dense_eval_all, 1, 16, label='Big-Dense', color='black', linestyle='dashdot')
# plt.xticks(x,x)
# plt.xlabel('Number of Experts')
# plt.ylabel('bpc')
# plt.title('4 Layers, 16 Experts')
# plt.legend(loc='upper right')
# plt.savefig('layer4_expert16_learnable.png', bbox_inches='tight')
# plt.close()
# # plt.show()


# plt.grid(linestyle='dashed', zorder=0)
# plt.rcParams['font.sans-serif'] = 'Times New Roman'
# plt.plot(x, rand, label='Random_MoE')
# plt.plot(x, rand_gradual, label='Random_MoE (Gradually)')
# plt.plot(x, rand_gradual_increase, label='Random_MoE (Gradually Decrease)')
# plt.plot(x, rand_early, label='Random_MoE (0)')
# plt.plot(x, rand_middle1, label='Random_MoE (1)')
# plt.plot(x, rand_late, label='Random_MoE (3)')
# plt.hlines(dense_eval_top2, 1, 16, label='Dense', color='black')
# plt.hlines(dense_eval_all, 1, 16, label='Big-Dense', color='black', linestyle='dashdot')
# plt.xticks(x,x)
# plt.xlabel('Number of Experts')
# plt.ylabel('bpc')
# plt.title('4 Layers, 16 Experts')
# plt.legend(loc='upper right')
# plt.savefig('layer4_expert16_random.png', bbox_inches='tight')
# plt.close()
# # plt.show()


# plt.grid(linestyle='dashed', zorder=0)
# plt.rcParams['font.sans-serif'] = 'Times New Roman'
# plt.plot(x, freeze, label='Learnable_MoE-Freeze')
# plt.plot(x, freeze_every, label='Learnable_MoE-Freeze (1,3)')
# plt.plot(x, freeze_early, label='Learnable_MoE-Freeze (0)')
# plt.plot(x, freeze_middle1, label='Learnable_MoE-Freeze (1)')
# plt.plot(x, freeze_middle2, label='Learnable_MoE-Freeze (2)')
# plt.plot(x, freeze_late, label='Learnable_MoE-Freeze (3)')
# plt.hlines(dense_eval_top2, 1, 16, label='Dense', color='black')
# plt.hlines(dense_eval_all, 1, 16, label='Big-Dense', color='black', linestyle='dashdot')
# plt.xticks(x,x)
# plt.xlabel('Number of Experts')
# plt.ylabel('bpc')
# plt.title('4 Layers, 16 Experts')
# plt.legend(loc='upper right')
# plt.savefig('layer4_expert16_freeze.png', bbox_inches='tight')
# plt.close()
# # plt.show()



plt.grid(linestyle='dashed', zorder=0)
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.plot(x, freeze, label='Learnable_MoE-Freeze')
plt.plot(x, rand, label='Random_MoE')
plt.plot(x, rand_dts_1, label='Random_MoE-DTS-Top1')
plt.plot(x, rand_dts_2, label='Random_MoE-DTS-Top2')
plt.plot(x, moe_dts_1, label='Learnable_MoE-DTS-Top1')
plt.plot(x, moe_dts_2, label='Learnable_MoE-DTS-Top2')
plt.hlines(dense_eval_top2, 1, 16, label='Dense', color='black')
plt.hlines(dense_eval_all, 1, 16, label='Big-Dense', color='black', linestyle='dashdot')
plt.xticks(x,x)
plt.xlabel('Number of Experts')
plt.ylabel('bpc')
plt.title('4 Layers, 16 Experts')
plt.legend(loc='upper right')
plt.savefig('layer4_expert16_dts.png', bbox_inches='tight')
plt.close()
# plt.show()