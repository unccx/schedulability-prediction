from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F
from dhg import Hypergraph
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import models.nn as nn

from .BasicModule import BasicModule

if TYPE_CHECKING:
    from data.dataset import LinkPredictDataset


class HGNNPSchedulabilityPredictor(BasicModule):
    def __init__(
        self,
        in_vertex_channels: int,
        hid_vertex_channels: int,
        out_vertex_channels: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.model_name = "HGNNP"
        self.hgconv1 = nn.HGNNPConv(
            in_vertex_channels, hid_vertex_channels, use_bn=use_bn, drop_rate=drop_rate
        )
        self.hgconv2 = nn.HGNNPConv(
            hid_vertex_channels, out_vertex_channels, use_bn=use_bn, is_last=True
        )
        self.v2e_msg_pass = nn.V2EMsgPass()
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(out_vertex_channels, 1), torch.nn.Sigmoid()
        )

    def forward(
        self, X: torch.Tensor, msgpass_H: torch.Tensor, link_pred_hyper_H: torch.Tensor
    ):
        X = self.hgconv1(X, msgpass_H)
        X = self.hgconv2(X, msgpass_H)
        hyperedge_embeddings = self.v2e_msg_pass(X, link_pred_hyper_H)
        score = self.predictor(hyperedge_embeddings)
        return score

    # def train_epoch(
    #     self, dataloader: DataLoader, criterion: Callable, optimizer: Optimizer
    # ) -> float:
    #     dataset: LinkPredictDataset = dataloader.dataset  # type: ignore
    #     loss_mean = 0
    #     for pos_hyperedges, neg_hyperedges in dataloader:
    #         pos_hg = Hypergraph(dataset.num_v, pos_hyperedges, device=dataset.device)
    #         neg_hg = Hypergraph(dataset.num_v, neg_hyperedges, device=dataset.device)
    #         optimizer.zero_grad()

    #         pos_scores = self.forward(
    #             dataset.X, dataset.msgpass_hg.H.to_dense(), pos_hg.H.to_dense()
    #         ).squeeze()
    #         neg_scores = self.forward(
    #             dataset.X, dataset.msgpass_hg.H.to_dense(), neg_hg.H.to_dense()
    #         ).squeeze()

    #         scores = torch.cat([pos_scores, neg_scores])
    #         labels = torch.cat(
    #             [torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]
    #         ).to(dataset.device)

    #         loss = criterion(scores, labels)  # 交叉熵损失
    #         loss.backward()
    #         optimizer.step()
    #         loss_mean += loss.item() * len(pos_hyperedges)

    #     loss_mean /= len(dataset)  # type: ignore
    #     return loss_mean
