import torch
import os

import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

def simple_moving_average(tensor, window_size):
    cumsum = torch.cumsum(tensor, dim=0)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

path = 'out/RTE/CoFi/RTE_sparsity0.60_lora_withffn/'

if __name__ == '__main__':
    final_zs = torch.load(os.path.join(path, 'zs.pt'), map_location='cpu')
    first_zs = torch.load(os.path.join(path, 'zs_tracking_1.pt'), map_location='cpu')
    total_head_zs = []
    for i in tqdm(range(1, 99)):
        zs = torch.load(os.path.join(path, 'zs_tracking_{}.pt'.format(i)), map_location='cpu')
        head_z = torch.stack([hz[:, 0, :, 0, 0] * hlz for hz, hlz in zip(zs['head_z'], zs['head_layer_z'])], dim=0)
        total_head_zs.append(head_z)
    
    total_head_zs = torch.cat(total_head_zs, dim=0)
    total_head_zs = total_head_zs.view(total_head_zs.shape[0], -1)
    
    plt.clf()
    for i in range(144):  
        selected_head_z = total_head_zs[:, i]
        # Apply moving average
        smoothed_selected_head_z = simple_moving_average(selected_head_z, 100)
        plt.plot(smoothed_selected_head_z.numpy())
    
    plt.ylim(0, 1)
    plt.savefig('head_zs_change.png')
    
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