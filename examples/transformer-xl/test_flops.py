import pytest
import torch
import torch.nn as nn
from thop import profile

from linear import FMoELinear

class TestUtils:
    def test_matmul_case2(self):
        n, in_c, out_c = 1, 100, 200
        net = nn.Linear(in_c, out_c)
        flops, params = profile(net, inputs=(torch.randn(n, in_c), ))
        print(flops, params)
        assert flops == n * in_c * out_c

    def test_matmul_case2(self):
        for i in range(10):
            n, in_c, out_c = torch.randint(1, 500, (3,)).tolist()
            net = nn.Linear(in_c, out_c)
            flops, params = profile(net, inputs=(torch.randn(n, in_c), ))
            print(flops, params)
            assert flops == n * in_c * out_c
    
    def test_conv2d(self):
        n, in_c, out_c = torch.randint(1, 500, (3,)).tolist()
        net = nn.Linear(in_c, out_c)
        flops, params = profile(net, inputs=(torch.randn(n, in_c), ))
        print(flops, params)
        assert flops == n * in_c * out_c



# class _Expert(nn.Module):

#     def __init__(self, num_expert, d_model, d_hidden):
#         super().__init__()
#         self.l1 = nn.Linear(d_model, d_hidden)
#         self.num_expert = num_expert

#     def forward(self, inp):
#         x = self.l1(inp)
#         return x

# def count_your_model(model)

# x = torch.rand()

# from linear import FMoELinear

# net = FMoELinear(16,256,512).cuda()
# x = torch.rand(2560,256).cuda()
# fwd_expert_count = torch.tensor([112,65,22,440,146,229,222,220,56,281,115,400,31,25,63,133]).cuda()


def calculate_linear(in_feature, num_elements):
    return torch.DoubleTensor([int(in_feature * num_elements)])

def count_flinear(m, x, y):
    # per output element
    total_mul = m.in_feat
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()

    m.total_ops += calculate_linear(total_mul, num_elements)

# flops, params = profile(net, inputs=(x, fwd_expert_count, ), custom_ops={FMoELinear: count_flinear})
# print(flops, params)