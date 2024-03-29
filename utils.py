import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from dhg import Hypergraph

from data import LinkPredictDataset


def calculate_sparsity(matrix):
    nonzero_elements = np.count_nonzero(matrix)
    total_elements = matrix.size

    nonzero_ratio = nonzero_elements / total_elements
    zero_ratio = 1 - nonzero_ratio

    print(f"非零元素比例：{nonzero_ratio:.2%}")
    print(f"零元素比例：{zero_ratio:.2%}")

# calculate_sparsity(HG.H.to_dense().cpu().numpy())

def calculate_system_utilization(dataset:LinkPredictDataset, scores, labels):
    """计算系统利用率"""
    y_pred = torch.where(scores >= scores.mean(), 1, 0)
    prediction_correct = torch.eq(y_pred, labels)
    prediction_error = ~prediction_correct

    correct_utilization_distribution = dataset.all_system_utilization_distribution[prediction_correct]
    error_utilization_distribution = dataset.all_system_utilization_distribution[prediction_error]

    return (dataset.all_system_utilization_distribution, 
            dataset.pos_hg_system_utilization_distribution, 
            dataset.neg_hg_system_utilization_distribution, 
            correct_utilization_distribution,
            error_utilization_distribution
            )

def compute_gradient_norm(net, pred):
    total_norm = 0
    parameters = [p for p in itertools.chain(net.parameters(), pred.parameters()) if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def collate(batch):
    transposed = list(zip(*batch))
    pos_hyperedge_list, neg_hyperedge_list = transposed
    return list(pos_hyperedge_list), list(neg_hyperedge_list)

def plot_dergee_utilization(hg:Hypergraph, title=r'Degree-Utilization'):
    # 绘制节点度数的图
    degree_values = hg.deg_v
    # plt.bar(range(len(degree_values)), degree_values)
    # plt.bar([index * 0.001 for index in range(len(degree_values))], degree_values)
    plt.scatter([index / len(degree_values) for index in range(len(degree_values))], degree_values)
    plt.xlabel('Utilization')
    plt.ylabel('Degree')
    plt.title(title)
    plt.show()

def zero_dergee_num(hg:Hypergraph):
    zero_dergee_num = 0
    for dergee in hg.deg_v:
        if dergee == 0:
            zero_dergee_num += 1
    return zero_dergee_num