import numpy as np
from fmoe.gates.base_gate import BaseGate

__all__ = ['set_top_k', 'set_router_mode', 'freeze_part_weight', 'adjust_moe_gate_number',
            'show_dts_gate_number', 'set_temperature', 'set_threshold']


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

    for name, m in model.named_modules():
        if hasattr(m, 'top_k') and hasattr(m, 'gate'):
            if isinstance(m.gate, BaseGate):
                if flag:
                    m.top_k = args.moe_num_expert
                else:
                    m.top_k = args.moe_top_k
                print('Set {}, Top-K = {}'.format(name, m.top_k))

def freeze_part_weight(model, args):
    if args.freeze_gate:
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






