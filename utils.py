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


def system_utilization_distribution(
    dataset: LinkPredictDataset, pos_scores: torch.Tensor, neg_scores: torch.Tensor
):
    """计算系统利用率"""
    scores = torch.cat([pos_scores, neg_scores])

    pos_y_pred = torch.where(pos_scores >= scores.mean(), True, False)
    neg_y_pred = torch.where(neg_scores < scores.mean(), True, False)
    pos_histogram = torch.histc(
        dataset.pos_hg_system_utilizations, bins=10, min=0, max=1
    )
    neg_histogram = torch.histc(
        dataset.neg_hg_system_utilizations, bins=10, min=0, max=1
    )

    true_positive_histogram = torch.histc(
        dataset.pos_hg_system_utilizations[pos_y_pred], bins=10, min=0, max=1
    )
    false_negative_histogram = torch.histc(
        dataset.pos_hg_system_utilizations[~pos_y_pred], bins=10, min=0, max=1
    )
    true_negative_histogram = torch.histc(
        dataset.neg_hg_system_utilizations[neg_y_pred], bins=10, min=0, max=1
    )
    false_positive_histogram = torch.histc(
        dataset.neg_hg_system_utilizations[~neg_y_pred], bins=10, min=0, max=1
    )

    # 在不同利用率区间可调度的比例
    schedulable_histogram = torch.histc(
        dataset.schedulable_hg_system_utilizations, bins=10, min=0, max=1
    )
    schedulable_ratio = schedulable_histogram / (schedulable_histogram + neg_histogram)
    print(f"schedulable_ratio: {schedulable_ratio}")

    # 在不同利用率区间的数据可调度的比例
    pos_ratio = (true_positive_histogram + false_negative_histogram) / (
        true_positive_histogram
        + false_positive_histogram
        + true_negative_histogram
        + false_negative_histogram
    )
    print(f"pos_ratio: {pos_ratio}")

    # # 在不同利用率区间正确预测为可调度的比例
    # true_positive_ratios = true_positive_histogram / (pos_histogram)
    # print(f"true_positive_ratios: {true_positive_ratios}")

    # # 在不同利用率区间错误预测为可调度的比例
    # false_positive_ratio = false_positive_histogram / pos_histogram
    # print(f"false_positive_ratio: {false_positive_ratio}")

    # # 在不同利用率区间正确预测为不可调度的比例
    # true_negative_ratio = true_negative_histogram / neg_histogram
    # print(f"true_negative_ratio: {true_negative_ratio}")

    # # 在不同利用率区间错误预测为不可调度的比例
    # false_negative_ratio = false_negative_histogram / neg_histogram
    # print(f"false_negative_ratio: {false_negative_ratio}")

    # Precision
    precision = true_positive_histogram / (
        true_positive_histogram + false_positive_histogram
    )
    print(f"precision: {precision}")

    # Recall
    recall = true_positive_histogram / (
        true_positive_histogram + false_negative_histogram
    )
    print(f"recall: {recall}")


def hyperedge_size_distribution(
    dataset: LinkPredictDataset, pos_scores: torch.Tensor, neg_scores: torch.Tensor
):
    """任务集基数分布"""
    bins = 4
    scores = torch.cat([pos_scores, neg_scores])

    pos_y_pred = torch.where(pos_scores >= scores.mean(), True, False)
    neg_y_pred = torch.where(neg_scores < scores.mean(), True, False)
    pos_histogram = torch.histc(dataset.pos_hg_hyperedge_size, bins=bins)
    neg_histogram = torch.histc(dataset.neg_hg_hyperedge_size, bins=bins)

    true_positive_histogram = torch.histc(
        dataset.pos_hg_hyperedge_size[pos_y_pred], bins=bins
    )
    false_negative_histogram = torch.histc(
        dataset.pos_hg_hyperedge_size[~pos_y_pred], bins=bins
    )
    true_negative_histogram = torch.histc(
        dataset.neg_hg_hyperedge_size[neg_y_pred], bins=bins
    )
    false_positive_histogram = torch.histc(
        dataset.neg_hg_hyperedge_size[~neg_y_pred], bins=bins
    )

    # 在不同任务集基数区间可调度的比例
    schedulable_histogram = torch.histc(
        dataset.schedulable_hg_system_utilizations, bins=bins
    )
    schedulable_ratio = schedulable_histogram / (schedulable_histogram + neg_histogram)
    print(f"schedulable_ratio: {schedulable_ratio}")

    # 在不同任务集基数区间的数据可调度的比例
    pos_ratio = (true_positive_histogram + false_negative_histogram) / (
        true_positive_histogram
        + false_positive_histogram
        + true_negative_histogram
        + false_negative_histogram
    )
    print(f"pos_ratio: {pos_ratio}")

    # Precision
    precision = true_positive_histogram / (
        true_positive_histogram + false_positive_histogram
    )
    print(f"precision: {precision}")

    # Recall
    recall = true_positive_histogram / (
        true_positive_histogram + false_negative_histogram
    )
    print(f"recall: {recall}")


def compute_gradient_norm(net, pred):
    total_norm = 0
    parameters = [
        p
        for p in itertools.chain(net.parameters(), pred.parameters())
        if p.grad is not None and p.requires_grad
    ]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def collate(batch):
    transposed = list(zip(*batch))
    pos_hyperedge_list, neg_hyperedge_list = transposed
    return list(pos_hyperedge_list), list(neg_hyperedge_list)


def plot_dergee_utilization(hg: Hypergraph, title=r"Degree-Utilization"):
    # 绘制节点度数的图
    degree_values = hg.deg_v
    # plt.bar(range(len(degree_values)), degree_values)
    # plt.bar([index * 0.001 for index in range(len(degree_values))], degree_values)
    plt.scatter(
        [index / len(degree_values) for index in range(len(degree_values))],
        degree_values,
    )
    plt.xlabel("Utilization")
    plt.ylabel("Degree")
    plt.title(title)
    plt.show()


def zero_dergee_num(hg: Hypergraph):
    zero_dergee_num = 0
    for dergee in hg.deg_v:
        if dergee == 0:
            zero_dergee_num += 1
    return zero_dergee_num
