from dhg import Hypergraph
import torch
import itertools
import numpy as np

def calculate_sparsity(matrix):
    nonzero_elements = np.count_nonzero(matrix)
    total_elements = matrix.size

    nonzero_ratio = nonzero_elements / total_elements
    zero_ratio = 1 - nonzero_ratio

    print(f"非零元素比例：{nonzero_ratio:.2%}")
    print(f"零元素比例：{zero_ratio:.2%}")

# calculate_sparsity(HG.H.to_dense().cpu().numpy())

def calculate_system_utilization(dataset, scores, labels):
    """计算系统利用率"""
    y_pred = torch.where(scores >= scores.mean(), 1, 0)
    prediction_correct = torch.eq(y_pred, labels)

    normalized_utilization = dataset.X.index_select(dim=1, index=torch.tensor(3).to(torch.device("cuda")))
    sum_of_normalized_utilization_pos = torch.sparse.mm(dataset.pos_hg.H_T, normalized_utilization)
    sum_of_normalized_utilization_neg = torch.sparse.mm(dataset.neg_hg.H_T, normalized_utilization)
    sum_of_normalized_utilization = torch.cat((sum_of_normalized_utilization_pos, sum_of_normalized_utilization_neg), dim=0)
    all_system_utilization = sum_of_normalized_utilization / dataset.S_m
    pos_utilization = sum_of_normalized_utilization_pos / dataset.S_m
    neg_utilization = sum_of_normalized_utilization_neg / dataset.S_m
    correct_utilization = all_system_utilization[prediction_correct]

    return all_system_utilization.squeeze(), pos_utilization.squeeze(), neg_utilization.squeeze(), correct_utilization.squeeze()

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