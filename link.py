# %%
import sys
sys.path.append("../DeepHypergraph/") 

# %%
import time
import random
import itertools
import tqdm
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dhg.nn import BPRLoss
from dhg import Hypergraph
from dhg.random import set_seed
from dhg.metrics import LinkPredictionEvaluator as Evaluator

from model import *
from utils import *
from data import LinkPredictDataset

# %%
def train(net, pred, data_loader, optimizer, epoch):
    global device, writer
    net.train()
    pred.train()

    st = time.time()
    dataset:LinkPredictDataset = data_loader.dataset
    loss_mean = 0

    for pos_hyperedge, neg_hyperedge in data_loader:
        pos_hg = Hypergraph(dataset.num_v, pos_hyperedge, device=device)
        neg_hg = Hypergraph(dataset.num_v, neg_hyperedge, device=device)
        optimizer.zero_grad()

        hyperedge_embedding = net(dataset.X, dataset.msg_pass_hg)
        pos_scores = pred(hyperedge_embedding, pos_hg).squeeze()
        neg_scores = pred(hyperedge_embedding, neg_hg).squeeze()
 
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat(
            [torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]
        ).to(device)

        # loss = (1 - pos_scores + neg_scores).clamp(min=0).mean()  # 间隔损失
        # loss = BPRLoss()(pos_scores, neg_scores)                  # 贝叶斯个性化排序损失
        loss = F.binary_cross_entropy(scores, labels)             # 交叉熵损失
        loss.backward()
        optimizer.step()
        loss_mean += loss.item() * len(pos_hyperedge)

    # # 记录在训练集上正例和负例的连接分数的分布
    # hyperedge_embedding = net(dataset.X, dataset.msg_pass_hg)
    # pos_scores = pred(hyperedge_embedding, dataset.pos_hg).squeeze()
    # neg_scores = pred(hyperedge_embedding, dataset.neg_hg).squeeze()
    # writer["train"].add_histogram("Score/pos", pos_scores, epoch)
    # writer["train"].add_histogram("Score/neg", neg_scores, epoch)

    # 计算梯度norm
    grad_norm = compute_gradient_norm(net, pred)
    writer["train"].add_scalar("Gradient_norm", grad_norm, epoch)

    # 记录训练Accuracy
    eval_res = evaluator.validate(labels, scores)
    writer["train"].add_scalar("Accuracy", eval_res, epoch)

    # print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    loss_mean /= len(data_loader.dataset)
    return loss_mean

# %%
@torch.no_grad()
def infer(net, pred, data_loader, test=False):
    net.eval()
    pred.eval()

    dataset = data_loader.dataset

    hyperedge_embedding = net(dataset.X, dataset.msg_pass_hg)
    pos_scores = pred(hyperedge_embedding, dataset.pos_hg).squeeze()
    neg_scores = pred(hyperedge_embedding, dataset.neg_hg).squeeze()

    global device, writer 
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat(
        [torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]
    ).to(device)

    global evaluator
    if not test:
        # 计算val_loss
        val_loss = F.binary_cross_entropy(scores, labels)
        eval_res = evaluator.validate(labels, scores)
        return eval_res, val_loss
    else:
        # 记录在测试集上正例和负例的连接分数的分布
        writer["test"].add_histogram("Score/pos", pos_scores, epoch)
        writer["test"].add_histogram("Score/neg", neg_scores, epoch)

        # 计算利用率
        (all_system_utilization, 
         pos_utilization, 
         neg_utilization, 
         correct_utilization, 
         error_utilization) = calculate_system_utilization(dataset, scores,labels)
        writer["test"].add_histogram("Utilization/all_system_utilization", all_system_utilization, bins='sturges') # 所有样本（无论分类是否正确）的利用率
        writer["test"].add_histogram("Utilization/pos_utilization", pos_utilization, bins='sturges') # 所有正样本（无论分类是否正确）的利用率
        writer["test"].add_histogram("Utilization/neg_utilization", neg_utilization, bins='sturges') # 所有负样本（无论分类是否正确）的利用率
        writer["test"].add_histogram("Utilization/correct_utilization", correct_utilization, bins='sturges') # 所有分类正确的样本的利用率
        writer["test"].add_histogram("Utilization/error_utilization", error_utilization, bins='sturges') # 所有分类错误的样本的利用率
        eval_res = evaluator.test(labels, scores)
        return eval_res

# %%

set_seed(2023)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

msg_pass_ratio:float = 0.5
train_set    = LinkPredictDataset("data_s7000_p5_t1000_hs11_e100000", ratio=(msg_pass_ratio, 1-msg_pass_ratio))
validate_set = LinkPredictDataset("data_s7001_p5_t1000_hs11_e100000", ratio=(msg_pass_ratio, 1-msg_pass_ratio))
test_set     = LinkPredictDataset("data_s7002_p5_t1000_hs11_e100000", ratio=(msg_pass_ratio, 1-msg_pass_ratio))

print("已加载数据")

# %%

evaluator = Evaluator(["auc", "accuracy", "f1_score"], validate_index=1)
epochs = 500

batch_sz = 256
# batch_sz = None

train_loader    = DataLoader(train_set, batch_size=batch_sz, shuffle=True, collate_fn=collate)
validate_loader = DataLoader(validate_set, batch_size=batch_sz, shuffle=False)
test_loader     = DataLoader(test_set, batch_size=batch_sz, shuffle=False)

in_channels = train_set.feat_dim
hid_channels = 256
out_channels = 128
# net = HGNNP(in_channels, hid_channels, out_channels, use_bn=True, drop_rate=0)
# net = UniGAT(in_channels, hid_channels, out_channels, use_bn=True, drop_rate=0, num_heads=4)
net = UniGCN(in_channels, hid_channels, out_channels, use_bn=True, drop_rate=0)
# net = HyperGCN(in_channels, hid_channels, out_channels, use_bn=True, drop_rate=0)
pred = ScorePredictor(out_channels)

num_params = sum(param.numel() for param in net.parameters()) + sum(param.numel() for param in pred.parameters())
print(f"模型参数量：{num_params}")

optimizer = optim.Adam(itertools.chain(net.parameters(), pred.parameters()))
# optimizer = optim.Adam(itertools.chain(net.parameters(), pred.parameters()), lr=3e-4, weight_decay=5e-4)
# optimizer = optim.SGD(itertools.chain(net.parameters(), pred.parameters()), lr=0.001, momentum=0.9)

# print("正在从cpu转移数据到gpu...")
net = net.to(device)
pred = pred.to(device)

# %%

print("正在训练模型...")

writer = {
    "train": SummaryWriter("./logs/train"),
    "validate": SummaryWriter("./logs/validate"),
    "test": SummaryWriter("./logs/test"),
}
best_state = None
best_epoch, best_val = 0, 0
with tqdm.trange(epochs) as tq:
    for epoch in tq:
        # train
        train_loss = train(net, pred, train_loader, optimizer, epoch)
        writer["train"].add_scalar("Loss", train_loss, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad(): 
                val_res, val_loss = infer(net, pred, validate_loader)
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
test_res = infer(net, pred, test_loader, test=True)
print(f"final result: epoch: {best_epoch}")
print(test_res)
[getattr(writer[key], "close")() for key in writer] # writer.close()