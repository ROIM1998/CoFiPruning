import torch

import seaborn as sns
from matplotlib import pyplot as plt

def simple_moving_average(tensor, window_size):
    cumsum = torch.cumsum(tensor, dim=0)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

if __name__ == '__main__':
    final_zs = torch.load('out/SST2/CoFi/SST2_sparsity0.60_lora_withffn/zs.pt', map_location='cpu')
    first_zs = torch.load('out/SST2/CoFi/SST2_sparsity0.60_lora_withffn/zs_tracking_{}.pt'.format(0), map_location='cpu')
    
    first_zs_stacked = {}
    first_zs_stacked['head_z'] = torch.stack(first_zs['head_z'], dim=0)
    first_zs_stacked['head_z'] = first_zs_stacked['head_z'].view(first_zs_stacked['head_z'].shape[0], 12, 12)
    first_zs_stacked['intermediate_z'] = torch.stack(first_zs['intermediate_z'], dim=0)
    first_zs_stacked['intermediate_z'] = first_zs_stacked['intermediate_z'].view(first_zs_stacked['intermediate_z'].shape[0], 12, 3072)
    first_zs_stacked['hidden_z'] = torch.stack(first_zs['hidden_z'], dim=0)
    first_zs_stacked['mlp_z'] = torch.stack(first_zs['mlp_z'], dim=0)
    first_zs_stacked['head_layer_z'] = torch.stack(first_zs['head_layer_z'], dim=0)
    
    head_z_vars = first_zs_stacked['head_z'][-1000:, :, :].var(dim=0)
    head_z_means = first_zs_stacked['head_z'][-1000:, :, :].mean(dim=0)
    head_z_results = (final_zs['head_z'] != 0).float().view(12, 12)
    head_layer_z_results = (final_zs['head_layer_z'] != 0).float().view(12)
    head_z_results = head_z_results * head_layer_z_results
    
    # Analyze the retained heads and pruned heads, compare their variance and mean