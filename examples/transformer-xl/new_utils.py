import copy
import torch
import numpy as np
import torch.nn as nn 
from fmoe.gates.base_gate import BaseGate

__all__ = ['set_top_k', 'set_router_mode', 'freeze_part_weight', 'adjust_moe_gate_number',
            'show_dts_gate_number', 'set_temperature', 'set_threshold', 
            'SWA_Average']


def set_top_k(model, num=2):
    for name, m in model.named_modules():
        if hasattr(m, 'top_k') and hasattr(m, 'gate'):
            if isinstance(m.gate, BaseGate):
                m.top_k = num
                m.gate.top_k = num
                print('Layer name: {}, Top-K = {}, {}'.format(name, m.top_k, m.gate.top_k))

def set_router_mode(model, args, flag=True):
    for name, m in model.named_modules():
        if isinstance(m, BaseGate):
            m.dense_moe_flag = flag 
            print('Layer name: {}, Average MoE = {}'.format(name, m.dense_moe_flag))

    current_gate = 0
    for name, m in model.named_modules():
        if hasattr(m, 'top_k') and hasattr(m, 'gate'):
            if isinstance(m.gate, BaseGate):
                if flag:
                    m.top_k = args.moe_num_expert
                    m.gate.top_k = args.moe_num_expert
                else:
                    m.top_k = args.moe_top_k
                    m.gate.top_k = args.moe_top_k
                current_gate = m.top_k
                print('Set {}, Top-K = {} {}'.format(name, m.top_k, m.gate.top_k))
    return current_gate



def freeze_part_weight(model, args):
    if args.freeze_gate:
        print('Freeze Router')
        for name, p in model.named_parameters():
            if 'gate.gate' in name:
                p.requires_grad = False

    if args.freeze_main_network:
        for name, p in model.named_parameters():
            if '.experts.' in name:
                p.requires_grad = False

    if args.freeze_main_network_all:
        for name, p in model.named_parameters():
            if 'word_emb.emb_layers' in name: continue
            if 'crit.out_layers' in name: continue 
            if 'layers.' in name:
                if not 'gate.gate' in name:
                    p.requires_grad = False

    for name, p in model.named_parameters():
        if p.requires_grad:
            print('* Trainable Parameters {}, shape = {}'.format(name, p.shape))
        else:
            print('* Freeze Parameters {}, shape = {}'.format(name, p.shape))

def calculate_gate_number(steps, args, overall_steps, min_experts, max_experts):
    if args.dynamic_moe_mode == 'linear_increase':
        number_experts = max_experts - min_experts
        gate_num = round(number_experts * steps / overall_steps) + min_experts
    elif args.dynamic_moe_mode == 'linear_decrease':
        number_experts = min_experts - max_experts
        gate_num = round(number_experts * steps / overall_steps) + max_experts
    elif args.dynamic_moe_mode == 'cosine_decrease':
        number_experts = max_experts - min_experts
        cosine_value = np.cos(np.pi * steps / (2 * overall_steps))
        gate_num = round(number_experts * cosine_value) + min_experts
    elif args.dynamic_moe_mode == 'cosine_increase':
        number_experts = min_experts - max_experts
        cosine_value = np.cos(np.pi * steps / (2 * overall_steps))
        gate_num = round(number_experts * cosine_value) + max_experts
    gate_num = np.clip(gate_num, min_experts, max_experts)
    return gate_num

def adjust_moe_gate_number(model, steps, args, current_gate):
    new_gate_num = calculate_gate_number(steps, args, args.dynamic_overall_steps, args.moe_top_k, args.moe_num_expert)
    if new_gate_num != current_gate:
        print('* Set New Top-k = {}'.format(new_gate_num))
        set_top_k(model, new_gate_num)
        current_gate = new_gate_num
    return current_gate


## Dense to Sparse
def show_dts_gate_number(model):
    for name, m in model.named_modules():
        if isinstance(m, BaseGate):
            mean_experts = m.sum_top_k / m.forward_n
            layer_temp = m.temperature
            layer_threshold = m.threshold
            print('* Mean-Experts = {:.0f}, Temperature = {:.4f}, Threshold = {:.4f}'.format(mean_experts, layer_temp, layer_threshold))

def set_temperature(model, iterations, all_iteration, max_temp, min_temp):
    temp = max_temp + iterations * (min_temp - max_temp) / all_iteration
    for name, m in model.named_modules():
        if isinstance(m, BaseGate):
            m.temperature = temp

def set_threshold(model, args):
    if args.gate_name == 'CustomDTSGate':
        print('* Set threshold for DTS Gate')
        for name, m in model.named_modules():
            if isinstance(m, BaseGate):
                m.threshold = args.threshold



## Weight Average
class SWA_Average(nn.Module):
    def __init__(self, model, t_start, t_end, device):
        super(SWA_Average, self).__init__()
        self.device = device
        self.average_model = copy.deepcopy(model) 
        self.register_buffer('n_average', torch.tensor(0, dtype=torch.long, device=self.device))
        self.t_start = t_start
        self.t_end = t_end 
    
    def forward(self, data, target, *mems):
        return self.average_model(data, target, *mems)
    
    def avg_fn(self, averaged_model_parameter, model_parameter, num_averaged):
        return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (
            num_averaged + 1
        )

    def update_parameters(self, current_model, step):
        if step >= self.t_start and step <= self.t_end:
            print('Update parameters with step {}, current_n_average = {}'.format(step, self.n_average))
            for p_swa, p_model in zip(self.average_model.parameters(), current_model.parameters()):
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model.detach(), self.n_average))
            self.n_average +=1 


