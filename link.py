# %%
import sys
sys.path.append("../DeepHypergraph/") 

# %%
import time
import random
from copy import deepcopy
import itertools
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dhg.nn import BPRLoss
from dhg import Hypergraph
from dhg.random import set_seed
from dhg.metrics import LinkPredictionEvaluator as Evaluator
from model import *

from utils import *

# %%
def train(net, pred, X, msg_pass_hg, pos_hg, neg_hg, optimizer, epoch):
    net.train()
    pred.train()

    st = time.time()
    optimizer.zero_grad()
    hyperedge_embedding = net(X, msg_pass_hg)
    pos_scores = pred(hyperedge_embedding, pos_hg).squeeze()
    neg_scores = pred(hyperedge_embedding, neg_hg).squeeze()

    global device, writer 
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat(
        [torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]
    ).to(device)

    eval_res = evaluator.validate(labels, scores)
    writer["train"].add_scalar("Accuracy", eval_res, epoch)

    if epoch % 50 == 0:
        writer["score"].add_histogram("Score/pos", pos_scores, epoch)
        writer["score"].add_histogram("Score/neg", neg_scores, epoch)

    #间隔损失
    # loss = (1 - pos_scores + neg_scores).clamp(min=0).mean()
    # 贝叶斯个性化排序损失
    # loss = BPRLoss()(pos_scores, neg_scores)
    # 交叉熵损失
    loss = F.binary_cross_entropy(scores, labels)
    loss.backward()

    # 计算梯度norm
    grad_norm = compute_gradient_norm(net, pred)
    writer["train"].add_scalar("Gradient_norm", grad_norm, epoch)

    optimizer.step()
    # print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()

# %%
@torch.no_grad()
def infer(net, pred, X, msg_pass_hg, pos_hg, neg_hg, test=False):
    net.eval()
    pred.eval()
    hyperedge_embedding = net(X, msg_pass_hg)
    pos_scores = pred(hyperedge_embedding, pos_hg).squeeze()
    neg_scores = pred(hyperedge_embedding, neg_hg).squeeze()

    global device
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat(
        [torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]
    ).to(device)

    # # 计算val_loss
    # loss = F.binary_cross_entropy(scores, labels)

    # # 计算利用率
    # system_utilization = calculate_system_utilization(X, scores, labels, pos_hg, neg_hg, validate_data["platform"])
    # writer["score"].add_histogram("System Utilization", system_utilization)

    global evaluator
    if not test:
        # 计算val_loss
        val_loss = F.binary_cross_entropy(scores, labels)
        eval_res = evaluator.validate(labels, scores)
        return eval_res, val_loss
    else:
        # 计算利用率
        all_system_utilization, pos_utilization, neg_utilization, correct_utilization = calculate_system_utilization(X, scores, labels, pos_hg, neg_hg, test_data["platform"])
        writer["utilization"].add_histogram("Utilization/all_system_utilization", all_system_utilization, bins='sturges') # 所有样本（无论分类是否正确）的利用率
        writer["utilization"].add_histogram("Utilization/pos_utilization", pos_utilization, bins='sturges') # 所有正样本（无论分类是否正确）的利用率
        writer["utilization"].add_histogram("Utilization/neg_utilization", neg_utilization, bins='sturges') # 所有负样本（无论分类是否正确）的利用率
        writer["utilization"].add_histogram("Utilization/correct_utilization", correct_utilization, bins='sturges') # 所有分类正确的样本的利用率
        
        eval_res = evaluator.test(labels, scores)
        return eval_res

# %%
import csv
from pathlib import Path

def load_data(file_path: Path, ratio: float=0.3, data_balance: bool=True):
    hyperedge_list = []
    neg_hyperedge_list = []
    with open(file_path / "hyperedges.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            # 读取每个超边的顶点列表，并将它们添加到 hyperedge_list 中
            hyperedge_list.append(row)
    
    # 确保超边顶点的id数值是整数形式，便于后续的处理和计算
    hyperedge_list = [[int(v) for v in edge] for edge in hyperedge_list]

    # 将超图的边分为消息传递边和监督边
    random.shuffle(hyperedge_list)
    split_pos = int(len(hyperedge_list) * ratio)
    # 获取前ratio比例的元素作为消息传递边的列表
    msg_pass_hyperedge_list = hyperedge_list[:split_pos]
    # 获取后ratio比例的元素作为监督边的列表
    pos_hyperedge_list = hyperedge_list[split_pos:]

    with open(file_path / "minimal_unschedulable_combinations.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            neg_hyperedge_list.append(row) 

    # 确保超边顶点的id数值是整数形式，便于后续的处理和计算
    neg_hyperedge_list = [[int(v) for v in edge] for edge in neg_hyperedge_list]

    with open(file_path / "task_quadruples.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        features = [list(map(float, row)) for row in reader]

    # 将数据转换为 Tensor
    features = torch.tensor(features)

    with open(file_path / "platform.csv", "r") as file:
        reader = csv.reader(file)
        processor_speeds = [float(speed) for id, speed in reader]

    # 处理监督边和负采样边数据不平衡的问题
    if data_balance:
        if len(neg_hyperedge_list) > len(pos_hyperedge_list):
            neg_hyperedge_list = random.sample(neg_hyperedge_list, len(pos_hyperedge_list))
        elif len(neg_hyperedge_list) < len(pos_hyperedge_list):
            pos_hyperedge_list = random.sample(pos_hyperedge_list, len(neg_hyperedge_list))

    msg_pass_data = {"hyperedge_list": msg_pass_hyperedge_list, "num_edges" : len(msg_pass_hyperedge_list)}
    pos_data      = {"hyperedge_list": pos_hyperedge_list, "num_edges" : len(pos_hyperedge_list)}
    neg_data      = {"hyperedge_list": neg_hyperedge_list, "num_edges" : len(neg_hyperedge_list)}

    return {
        "msg_pass": msg_pass_data, 
        "pos": pos_data, 
        "neg": neg_data, 
        "vertices_feature": features, 
        "num_vertices": features.shape[0],
        "platform": processor_speeds
    }

# %%
set_seed(2023)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

data_balance = True
train_data =    load_data(Path("../EDF/data/data_s5000_p3_t1000_hs7_e20000/"), data_balance=data_balance)
validate_data = load_data(Path("../EDF/data/data_s5001_p3_t1000_hs7_e20000/"), data_balance=data_balance)
test_data =     load_data(Path("../EDF/data/data_s5002_p3_t1000_hs7_e20000/"), data_balance=data_balance)

print("已加载数据")

train_X = train_data["vertices_feature"]
train_msg_pass_HG = Hypergraph(train_data["num_vertices"], train_data["msg_pass"]["hyperedge_list"])
train_pos_HG      = Hypergraph(train_data["num_vertices"], train_data["pos"]["hyperedge_list"])
train_neg_HG      = Hypergraph(train_data["num_vertices"], train_data["neg"]["hyperedge_list"])

validate_X = validate_data["vertices_feature"]
validate_msg_pass_HG = Hypergraph(validate_data["num_vertices"], validate_data["msg_pass"]["hyperedge_list"])
validate_pos_HG      = Hypergraph(validate_data["num_vertices"], validate_data["pos"]["hyperedge_list"])
validate_neg_HG      = Hypergraph(validate_data["num_vertices"], validate_data["neg"]["hyperedge_list"])

test_X = test_data["vertices_feature"]
test_msg_pass_HG = Hypergraph(test_data["num_vertices"], test_data["msg_pass"]["hyperedge_list"])
test_pos_HG      = Hypergraph(test_data["num_vertices"], test_data["pos"]["hyperedge_list"])
test_neg_HG      = Hypergraph(test_data["num_vertices"], test_data["neg"]["hyperedge_list"])

# 不使用任务属性作为节点嵌入向量的初始化
# train_X = torch.eye(train_data["num_vertices"])
# validate_X = torch.eye(validate_data["num_vertices"])
# test_X = torch.eye(test_data["num_vertices"])

print("已建立超图")

# %%

evaluator = Evaluator(["auc", "accuracy", "f1_score"], validate_index=1)
epochs = 4000

in_channels = train_X.shape[1]
hid_channels = 512
out_channels = 256
net = HGNNP(in_channels, hid_channels, out_channels, use_bn=True, drop_rate=0)
# net = UniGAT(in_channels, hid_channels, out_channels, use_bn=True, drop_rate=0, num_heads=8)
pred = ScorePredictor(out_channels)

num_params = sum(param.numel() for param in net.parameters()) + sum(param.numel() for param in pred.parameters())
print(f"模型参数量：{num_params}")

optimizer = optim.Adam(itertools.chain(net.parameters(), pred.parameters()))
# optimizer = optim.Adam(itertools.chain(net.parameters(), pred.parameters()), lr=3e-4, weight_decay=5e-4)
# optimizer = optim.SGD(itertools.chain(net.parameters(), pred.parameters()), lr=0.001, momentum=0.9)

# %%

print("正在从cpu转移数据到gpu...")

net = net.to(device)
pred = pred.to(device)

train_X = train_X.to(device)
train_msg_pass_HG = train_msg_pass_HG.to(device)
train_pos_HG      = train_pos_HG.to(device)
train_neg_HG      = train_neg_HG.to(device)

validate_X = validate_X.to(device)
validate_msg_pass_HG = validate_msg_pass_HG.to(device)
validate_pos_HG      = validate_pos_HG.to(device)
validate_neg_HG      = validate_neg_HG.to(device)

test_X = test_X.to(device)
test_msg_pass_HG = test_msg_pass_HG.to(device)
test_pos_HG      = test_pos_HG.to(device)
test_neg_HG      = test_neg_HG.to(device)

# %%

print("转移完成")

print(f"X: {train_X.device}")
print(f"validate_X: {validate_X.device}")
print(f"test_X: {test_X.device}")

print(f"msg_pass_HG: {train_msg_pass_HG.device}")
print(f"validate_msg_pass_HG: {validate_msg_pass_HG.device}")
print(f"test_msg_pass_HG: {test_msg_pass_HG.device}")

print(f"pos_HG: {train_pos_HG.device}")
print(f"validate_pos_HG: {validate_pos_HG.device}")
print(f"test_pos_HG: {test_pos_HG.device}")

print(f"neg_HG: {train_neg_HG.device}")
print(f"validate_neg_HG: {validate_neg_HG.device}")
print(f"test_neg_HG: {test_neg_HG.device}")

print(f"net: {next(net.parameters()).device}")
print(f"pred: {next(pred.parameters()).device}")

# %%

print("正在训练模型...")

writer = {
    "train": SummaryWriter("./logs/train"),
    "validate": SummaryWriter("./logs/validate"),
    "score": SummaryWriter("./logs/score"),
    "utilization": SummaryWriter("./logs/utilization")
}
best_state = None
best_epoch, best_val = 0, 0
with tqdm.trange(epochs) as tq:
    for epoch in tq:
        # train
        train_loss = train(net, pred, train_X, train_msg_pass_HG, train_pos_HG, train_neg_HG, optimizer, epoch)
        writer["train"].add_scalar("Loss", train_loss, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad(): 
                val_res, val_loss = infer(net, pred, validate_X, validate_msg_pass_HG, validate_pos_HG, validate_neg_HG)
                writer["validate"].add_scalar("Loss", val_loss, epoch)
                writer["validate"].add_scalar("Accuracy", val_res, epoch)
            if val_res > best_val:
                # print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
        tq.set_postfix(
            {
                "best_epoch": f"{best_epoch}",
                "best_val": f"{best_val:.5f}",
                "train_loss": f"{train_loss:.5f}",
                "val_loss": f"{val_loss:.5f}",
                "validate_res": f"{val_res:.5f}",
            },
            refresh=False,
        )

[getattr(writer[key], "flush")() for key in writer] # writer.flush()
print("\ntrain finished!")
print(f"best val: {best_val:.5f}")

# %%
# test
print("test...")
if not (best_val == 0 or best_state):
    net.load_state_dict(best_state)
test_res = infer(net, pred, test_X, test_msg_pass_HG, test_pos_HG, test_neg_HG, test=True)
print(f"final result: epoch: {best_epoch}")
print(test_res)
[getattr(writer[key], "close")() for key in writer] # writer.close()