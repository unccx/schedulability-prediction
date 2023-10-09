import numpy as np
from dhg import Hypergraph
import torch
import matplotlib.pyplot as plt

def calculate_sparsity(matrix):
    nonzero_elements = np.count_nonzero(matrix)
    total_elements = matrix.size

    nonzero_ratio = nonzero_elements / total_elements
    zero_ratio = 1 - nonzero_ratio

    print(f"非零元素比例：{nonzero_ratio:.2%}")
    print(f"零元素比例：{zero_ratio:.2%}")

# calculate_sparsity(HG.H.to_dense().cpu().numpy())

def calculate_system_utilization(X, scores, labels, pos_hg:Hypergraph, neg_hg:Hypergraph, platform):
    """计算系统利用率"""
    fastest_processor_speed = platform[0]
    platform_speed = [processor_speed / fastest_processor_speed for processor_speed in platform] # 处理器速度归一化
    sum_of_normalized_speed = sum(platform_speed) # S_m

    y_pred = torch.where(scores >= scores.mean(), 1, 0)
    prediction_correct = torch.eq(y_pred, labels)

    normalized_utilization = X.index_select(dim=1, index=torch.tensor(3).to(torch.device("cuda")))
    sum_of_normalized_utilization_pos = torch.sparse.mm(pos_hg.H_T, normalized_utilization)
    sum_of_normalized_utilization_neg = torch.sparse.mm(neg_hg.H_T, normalized_utilization)
    sum_of_normalized_utilization = torch.cat((sum_of_normalized_utilization_pos, sum_of_normalized_utilization_neg), dim=0)
    all_system_utilization = sum_of_normalized_utilization / sum_of_normalized_speed
    pos_utilization = sum_of_normalized_utilization_pos / sum_of_normalized_speed
    neg_utilization = sum_of_normalized_utilization_neg / sum_of_normalized_speed
    correct_utilization = all_system_utilization[prediction_correct]

    return all_system_utilization.squeeze(), pos_utilization.squeeze(), neg_utilization.squeeze(), correct_utilization.squeeze()