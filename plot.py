import torch

import seaborn as sns
from matplotlib import pyplot as plt

def simple_moving_average(tensor, window_size):
    cumsum = torch.cumsum(tensor, dim=0)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

if __name__ == '__main__':
    final_zs = torch.load('out/SST2/CoFi/SST2_sparsity0.60_lora_withffn/zs.pt', map_location='cpu')
    zs_topk = [torch.load('out/SST2/CoFi/SST2_sparsity0.60_lora_withffn/zs_tracking_{}.pt'.format(i), map_location='cpu') for i in range(19)]
    all_head_zs = torch.stack([v for zs in zs_topk for v in zs['head_z']], dim=0)
    all_head_zs = all_head_zs.view(all_head_zs.shape[0], 144)
    plt.clf()
    
    for i in range(144):  
        selected_head_z = all_head_zs[:, i]
        # Apply moving average
        smoothed_selected_head_z = simple_moving_average(selected_head_z, 100)
        plt.plot(smoothed_selected_head_z.numpy())
    
    plt.ylim(0, 1)
    plt.savefig('head_score.png')
    
    
    all_intermediate_zs = all_head_zs = torch.stack([v for zs in zs_topk for v in zs['intermediate_z']], dim=0)
    all_intermediate_zs = all_intermediate_zs.view(all_intermediate_zs.shape[0], 12, 3072)
    
    plt.clf()
    for i in range(100):
        selected_intermediate_z = all_intermediate_zs[:, 0, i]
        smoothed_selected_intermediate_z = simple_moving_average(selected_intermediate_z, 100)
        plt.plot(smoothed_selected_intermediate_z.numpy())
    
    plt.ylim(0, 1)
    plt.savefig('intermediate_score.png')
    
    # Plot coarse-grained mlp_layer
    all_head_layer_zs = torch.stack([v for zs in zs_topk for v in zs['head_layer_z']], dim=0)
    plt.clf()
    for i in range(12):
        selected_head_layer_z = all_head_layer_zs[:, i]
        # Apply moving average
        smoothed_selected_head_layer_z = simple_moving_average(selected_head_layer_z, 100)
        plt.plot(smoothed_selected_head_layer_z.numpy())
    
    plt.ylim(0, 1)
    plt.savefig('head_layer_score.png')