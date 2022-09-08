import torch


def pad_sequence_reverse(data):
    # data should be a list of 1D tensors

    assert data[0].dim() == 1
    device = data[0].device
    length_list = []
    for item in data:
        length_list.append(item.shape[0])
    max_length = max(length_list)

    # padding 
    padded_data_list = []
    for item in data:
        padded_item = torch.cat([torch.zeros(max_length - item.shape[0]).to(device), item]).reshape(-1, 1)
        padded_data_list.append(padded_item)
    padded_data_list = torch.cat(padded_data_list, dim=1)
    return padded_data_list






