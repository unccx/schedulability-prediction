# %%
import sys
sys.path.append("../DeepHypergraph/") 

# %%
import time
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import dhg
from dhg import Hypergraph
from dhg.nn import HGNNPConv
from dhg.models import HGNNPLinkPred
from dhg.random import set_seed
from dhg.metrics import LinkPredictionEvaluator as Evaluator

# %%
import numpy as np

def calculate_sparsity(matrix):
    nonzero_elements = np.count_nonzero(matrix)
    total_elements = matrix.size

    nonzero_ratio = nonzero_elements / total_elements
    zero_ratio = 1 - nonzero_ratio

    print(f"非零元素比例：{nonzero_ratio:.2%}")
    print(f"零元素比例：{zero_ratio:.2%}")

# calculate_sparsity(HG.H.to_dense().cpu().numpy())

# %%
def train(net, X, hypergraph, negative_hypergraph, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    pos_score = net(X, hypergraph)
    neg_score = net(X, negative_hypergraph)

    scores = torch.cat([pos_score, neg_score]).squeeze()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).to(device)

    loss = F.binary_cross_entropy(scores, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()

# %%
@torch.no_grad()
def infer(net, X, hypergraph, negative_hypergraph, test=False):
    net.eval()
    pos_score = net(X, hypergraph)
    neg_score = net(X, negative_hypergraph)

    scores = torch.cat([pos_score, neg_score]).squeeze()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).to(device)

    if not test:
        res = evaluator.validate(labels, scores)
    else:
        res = evaluator.test(labels, scores)
    return res

# %%
import csv
from pathlib import Path

def load_data(file_path: Path):
    hyperedge_list = []
    neg_hyperedge_list = []
    with open(file_path / "hyperedges.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            # 读取每个超边的顶点列表，并将它们添加到 hyperedge_list 中
            hyperedge_list.append(row)
    
    hyperedge_list = [[int(v) for v in edge] for edge in hyperedge_list]

    with open(file_path / "negative_samples.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            neg_hyperedge_list.append(row) 

    neg_hyperedge_list = [[int(v) for v in edge] for edge in neg_hyperedge_list]

    with open(file_path / "task_quadruples.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = [list(map(float, row)) for row in reader]

    # 将数据转换为 Tensor
    features = torch.tensor(data)

    data = {"hyperedge_list": hyperedge_list, "num_edges" : len(hyperedge_list)}
    neg_data = {"hyperedge_list": neg_hyperedge_list, "num_edges" : len(neg_hyperedge_list)}

    return {"pos":data, "neg": neg_data, "vertices_feature" : features, "num_vertices" : features.shape[0]}

# %%
set_seed(2021)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
evaluator = Evaluator(["auc", "accuracy", "f1_score"], validate_index=1)

train_data =    load_data(Path("../EDF/data/data_s3000_p3_t1000_hs7_e10000/"))
validate_data = load_data(Path("../EDF/data/data_s3001_p3_t1000_hs7_e10000/"))
test_data =     load_data(Path("../EDF/data/data_s3002_p9_t1000_hs7_e5000/"))

print("已加载数据")

train_X = train_data["vertices_feature"]
# train_X = torch.eye(train_data["num_vertices"]) # 不使用任务属性作为节点嵌入向量的初始化
train_HG = Hypergraph(train_data["num_vertices"], train_data["pos"]["hyperedge_list"])
train_neg_HG = Hypergraph(train_data["num_vertices"], train_data["neg"]["hyperedge_list"])

# 随机采样部分超边建图
# ratio = 0.1
# train_pos_hyperedge_list = random.sample(train_data["pos"]["hyperedge_list"], int(train_data["pos"]["num_edges"] * ratio))
# train_neg_hyperedge_list = random.sample(train_data["neg"]["hyperedge_list"], int(train_data["neg"]["num_edges"] * ratio))
# train_HG = Hypergraph(train_data["num_vertices"], train_pos_hyperedge_list)
# train_neg_HG = Hypergraph(train_data["num_vertices"], train_neg_hyperedge_list)

validate_X = validate_data["vertices_feature"]
# validate_X = torch.eye(validate_data["num_vertices"]) # 不使用任务属性作为节点嵌入向量的初始化
validate_HG = Hypergraph(validate_data["num_vertices"], validate_data["pos"]["hyperedge_list"])
validate_neg_HG = Hypergraph(validate_data["num_vertices"], validate_data["neg"]["hyperedge_list"])

test_X = test_data["vertices_feature"]
# test_X = torch.eye(test_data["num_vertices"]) # 不使用任务属性作为节点嵌入向量的初始化
test_HG = Hypergraph(test_data["num_vertices"], test_data["pos"]["hyperedge_list"])
test_neg_HG = Hypergraph(test_data["num_vertices"], test_data["neg"]["hyperedge_list"])

print("已建立超图")
# print(f"train_HG: 超边数量{len(train_pos_hyperedge_list)}")
# print(f"train_neg_HG: 超边数量{len(train_neg_hyperedge_list)}")

# %%
net = HGNNPLinkPred(train_X.shape[1], 64, 32, use_bn=True)

# %%
optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=5e-4)

# %%

print("正在从cpu转移数据到gpu...")

train_X = train_X.to(device)
train_HG = train_HG.to(device)
train_neg_HG = train_neg_HG.to(device)
net = net.to(device)

validate_X = validate_X.to(device)
validate_HG = validate_HG.to(device)
validate_neg_HG = validate_neg_HG.to(device)

test_X = test_X.to(device)
test_HG = test_HG.to(device)
test_neg_HG = test_neg_HG.to(device)

# %%

print("转移完成")

print(f"X: {train_X.device}")
print(f"validate_X: {validate_X.device}")
print(f"test_X: {test_X.device}")

print(f"HG: {train_HG.device}")
print(f"validate_HG: {validate_HG.device}")
print(f"test_HG: {test_HG.device}")

print(f"neg_HG: {train_neg_HG.device}")
print(f"validate_neg_HG: {validate_neg_HG.device}")
print(f"test_neg_HG: {test_neg_HG.device}")

print(f"net: {next(net.parameters()).device}")

# %%

print("正在训练模型...")

best_state = None
best_epoch, best_val = 0, 0
for epoch in range(20000):
    # train
    train(net, train_X, train_HG, train_neg_HG, optimizer, epoch)
    # validation
    if epoch % 1 == 0:
        with torch.no_grad(): 
            val_res = infer(net, validate_X, validate_HG, validate_neg_HG)
        if val_res > best_val:
            print(f"update best: {val_res:.5f}")
            best_epoch = epoch
            best_val = val_res
            best_state = deepcopy(net.state_dict())
print("\ntrain finished!")
print(f"best val: {best_val:.5f}")


# %%
# test
print("test...")
if not (best_val == 0 or best_state):
    net.load_state_dict(best_state)
res = infer(net, test_X, test_HG, test_neg_HG, test=True)
print(f"final result: epoch: {best_epoch}")
print(res)


