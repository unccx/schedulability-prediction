from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F
import tqdm
from dhg import Hypergraph
from dhg.random import set_seed
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore

import models
from config import CONFIG
from data.dataset import LinkPredictDataset
from utils import collate, compute_gradient_norm
from utils.metric import LinkPredictionEvaluator

if TYPE_CHECKING:
    from models.BasicModule import BasicModule


@torch.no_grad()
def test(**kwargs):
    CONFIG.parse(kwargs)  # type: ignore

    # step1: data
    device = torch.device("cuda") if CONFIG.use_gpu else torch.device("cpu")
    test_data = LinkPredictDataset(
        dataset_path=CONFIG.test_data_root,
        msgpass_ratio=CONFIG.msgpass_ratio,
        device=device,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers,
    )

    # step2: config model
    model: BasicModule = getattr(models, CONFIG.model)(
        test_data.feat_dim, CONFIG.hid_channels, CONFIG.out_channels, CONFIG.use_bn
    )
    if CONFIG.load_model_path:
        model.load(CONFIG.load_model_path)
    model.to(device)

    # step3: test
    model.eval()
    pos_scores = model(
        test_data.X, test_data.msgpass_hg.H.to_dense(), test_data.pos_hg.H.to_dense()
    ).squeeze()
    neg_scores = model(
        test_data.X, test_data.msgpass_hg.H.to_dense(), test_data.neg_hg.H.to_dense()
    ).squeeze()
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat(
        [torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]
    ).to(device)

    evaluator = LinkPredictionEvaluator(
        ["auc", "accuracy", "f1_score", "confusion_matrix"], validate_index=1
    )
    test_res = evaluator.test(labels, scores)  # type: ignore
    return test_res


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    evaluator: LinkPredictionEvaluator,
):
    # 把模型设为验证模式
    model.eval()

    dataset: LinkPredictDataset = dataloader.dataset  # type: ignore
    pos_scores = model(
        dataset.X, dataset.msgpass_hg.H.to_dense(), dataset.pos_hg.H.to_dense()
    ).squeeze()
    neg_scores = model(
        dataset.X, dataset.msgpass_hg.H.to_dense(), dataset.neg_hg.H.to_dense()
    ).squeeze()
    scores = torch.cat([pos_scores, neg_scores]).to(dataset.device)
    labels = torch.cat(
        [torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]
    ).to(dataset.device)

    # 计算val_loss
    val_loss = criterion(scores, labels)
    eval_res = evaluator.validate(labels, scores)  # type: ignore

    # 把模型恢复为训练模式
    model.train()

    return eval_res, val_loss


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    optimizer: Optimizer,
) -> float:
    dataset: LinkPredictDataset = dataloader.dataset  # type: ignore
    loss_mean = 0
    for pos_hg_H, neg_hg_H in dataloader:
        pos_hg_H, neg_hg_H = pos_hg_H.to(dataset.device), neg_hg_H.to(dataset.device)

        optimizer.zero_grad()

        pos_scores = model(
            dataset.X, dataset.msgpass_hg.H.to_dense(), pos_hg_H
        ).squeeze()
        neg_scores = model(
            dataset.X, dataset.msgpass_hg.H.to_dense(), neg_hg_H
        ).squeeze()

        scores = torch.cat([pos_scores, neg_scores]).to(dataset.device)
        labels = torch.cat(
            [torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]
        ).to(dataset.device)

        loss = criterion(scores, labels)  # 交叉熵损失
        loss.backward()
        optimizer.step()
        loss_mean += loss.item() * CONFIG.batch_size

    loss_mean /= len(dataset)  # type: ignore
    return loss_mean


def train(**kwargs):
    CONFIG.parse(kwargs)  # type: ignore

    # step1: data
    device = torch.device("cuda") if CONFIG.use_gpu else torch.device("cpu")
    print("加载数据中...")
    train_data = LinkPredictDataset(
        dataset_path=CONFIG.train_data_root,
        msgpass_ratio=CONFIG.msgpass_ratio,
        device=device,
    )
    val_data = LinkPredictDataset(
        dataset_path=CONFIG.val_data_root,
        msgpass_ratio=CONFIG.msgpass_ratio,
        device=device,
    )
    train_dataloader = DataLoader(
        train_data,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        collate_fn=partial(collate, num_v=train_data.num_v, device=device),
        num_workers=CONFIG.num_workers,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers,
    )
    print("已加载数据")

    # step2: config model
    set_seed(CONFIG.seed)
    model: BasicModule = getattr(models, CONFIG.model)(
        train_data.feat_dim, CONFIG.hid_channels, CONFIG.out_channels, CONFIG.use_bn
    )
    if CONFIG.load_model_path:
        model.load(CONFIG.load_model_path)
    model.to(device)

    # step3: criterion and optimizer
    criterion = F.binary_cross_entropy
    lr = CONFIG.lr
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=CONFIG.weight_decay
    )

    # step4: config log
    writer = {
        "train": SummaryWriter("./logs/train"),
        "validate": SummaryWriter("./logs/validate"),
        # "test": SummaryWriter("./logs/test"),
    }
    evaluator = LinkPredictionEvaluator(
        ["auc", "accuracy", "f1_score", "confusion_matrix"], validate_index=1
    )

    # step5: train
    model_best_state = None
    best_epoch, best_val = 0, 0
    epoch_iterator = tqdm.trange(CONFIG.max_epoch)
    for epoch in epoch_iterator:
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer)
        val_res, val_loss = validate(model, val_dataloader, criterion, evaluator)

        writer["validate"].add_scalar("Loss", val_loss, epoch)
        writer["validate"].add_scalar("Accuracy", val_res, epoch)
        # writer["train"].add_scalar("Gradient norm", compute_gradient_norm(model), epoch)
        [getattr(writer[key], "flush")() for key in writer]  # writer.flush()

        if val_res > best_val:
            best_epoch = epoch
            best_val = val_res
            model_best_state = deepcopy(model.state_dict())

        epoch_iterator.set_postfix(
            {
                "best_epoch": f"{best_epoch}",
                "best_val": f"{best_val:.5f}",
                "train_loss": f"{train_loss:.5f}",
                "val_loss": f"{val_loss:.5f}",
                "validate_res": f"{val_res:.5f}",
            },
            refresh=False,
        )

    [getattr(writer[key], "close")() for key in writer]  # writer.close()

    if model_best_state is not None:
        model.load_state_dict(model_best_state)
        CONFIG.load_model_path = model.save()

    return CONFIG.load_model_path


def help():
    """
    打印帮助的信息： python main.py help
    """

    print(
        """
    usage : python {0} <function> [--args=value,]
    <function> := train | test | help
    example: 
            python {0} train --lr=0.01
            python {0} test --test_data_root='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(
            __file__
        )
    )

    from inspect import getsource

    source = getsource(CONFIG.__class__)
    print(source)


if __name__ == "__main__":
    import fire

    fire.Fire()
