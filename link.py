# %%
import itertools
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from dhg import Hypergraph
from dhg.nn import BPRLoss
from dhg.random import set_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from data import LinkPredictDataset
from metric import LinkPredictionEvaluator as Evaluator
from model import *
from utils import *


# %%
def train(net, pred, data_loader, optimizer, epoch):
    global device, writer
    net.train()
    pred.train()

    st = time.time()
    dataset: LinkPredictDataset = data_loader.dataset
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
        loss = F.binary_cross_entropy(scores, labels)  # 交叉熵损失
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
    eval_res = evaluator.validate(labels, scores)  # type: ignore
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

    start = time.time()
    hyperedge_embedding = net(dataset.X, dataset.msg_pass_hg)
    pos_scores = pred(hyperedge_embedding, dataset.pos_hg).squeeze()
    neg_scores = pred(hyperedge_embedding, dataset.neg_hg).squeeze()
    end = time.time()

    global device, writer
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat(
        [torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]
    ).to(device)

    global evaluator
    if not test:
        # 计算val_loss
        val_loss = F.binary_cross_entropy(scores, labels)
        eval_res = evaluator.validate(labels, scores)  # type: ignore
        return eval_res, val_loss
    else:
        # 记录在测试集上正例和负例的连接分数的分布
        writer["test"].add_histogram("Score/pos", pos_scores, epoch)
        writer["test"].add_histogram("Score/neg", neg_scores, epoch)

        # 在不同利用率区间中的性能差异
        system_utilization_distribution(dataset, pos_scores, neg_scores)
        print("————————————————————————————————————————————————————————————")
        # 在不同任务集基数区间中的性能差异
        hyperedge_size_distribution(dataset, pos_scores, neg_scores)

        print(f"Inference latency: {epoch}, Time: {end-start:.5f}sec.")

        eval_res = evaluator.test(labels, scores)  # type: ignore
        return eval_res


# %%

set_seed(2023)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("加载数据中...")

msg_pass_ratio: float = 0.7
train_set = LinkPredictDataset(
    dataset_name="2024-10-15_22-37-01",
    root_dir=Path("./data"),
    ratio=(msg_pass_ratio, 1 - msg_pass_ratio),
)
# msg_pass_ratio = 0.5
validate_set = LinkPredictDataset(
    dataset_name="2024-10-16_10-14-09",
    root_dir=Path("./data"),
    ratio=(msg_pass_ratio, 1 - msg_pass_ratio),
    # data_balance=False,
)
# msg_pass_ratio = 0.5
test_set = LinkPredictDataset(
    dataset_name="2024-10-16_18-13-06",
    root_dir=Path("./data"),
    ratio=(msg_pass_ratio, 1 - msg_pass_ratio),
    # data_balance=False,
)

print("已加载数据")

# %%

evaluator = Evaluator(
    ["auc", "accuracy", "f1_score", "confusion_matrix"], validate_index=1
)
epochs = 100

batch_sz = 8

train_loader = DataLoader(
    train_set, batch_size=batch_sz, shuffle=True, collate_fn=collate
)
validate_loader = DataLoader(validate_set, batch_size=batch_sz, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_sz, shuffle=False)

in_channels = train_set.feat_dim
hid_channels = 1024
out_channels = 1024
net = HGNNP(in_channels, hid_channels, out_channels, use_bn=True, drop_rate=0.5)
# net = UniGAT(
#     in_channels, hid_channels, out_channels, use_bn=True, drop_rate=0, num_heads=4
# )
# net = UniGCN(in_channels, hid_channels, out_channels, use_bn=True, drop_rate=0)
# net = HyperGCN(in_channels, hid_channels, out_channels, use_bn=True, drop_rate=0)
pred = ScorePredictor(out_channels)

num_params = sum(param.numel() for param in net.parameters()) + sum(
    param.numel() for param in pred.parameters()
)
print(f"模型参数量：{num_params}")

# optimizer = optim.Adam(itertools.chain(net.parameters(), pred.parameters()))
optimizer = optim.Adam(
    itertools.chain(net.parameters(), pred.parameters()), lr=0.001, weight_decay=5e-4
)
# optimizer = optim.SGD(
#     itertools.chain(net.parameters(), pred.parameters()), lr=0.001, momentum=0.9
# )

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
net_best_state, pred_best_state = None, None
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
                net_best_state = deepcopy(net.state_dict())
                pred_best_state = deepcopy(pred.state_dict())
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

[getattr(writer[key], "flush")() for key in writer]  # writer.flush()
print("\ntrain finished!")
print(f"best val: {best_val:.5f}")

# %%
# test
print("test...")
if not (best_val == 0 or net_best_state or pred_best_state):
    net.load_state_dict(net_best_state)  # type: ignore
    net.load_state_dict(pred_best_state)  # type: ignore
test_res = infer(net, pred, test_loader, test=True)
print(f"final result: epoch: {best_epoch}")
print(test_res)
[getattr(writer[key], "close")() for key in writer]  # writer.close()

torch.save(net, Path("./net.pth"))
torch.save(pred, Path("./pred.pth"))
