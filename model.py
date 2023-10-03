import torch
import torch.nn as nn

import dhg
from dhg.nn import HGNNPConv

class ScorePredictor(nn.Module):
    def __init__(self, embedding_dimension):
        super().__init__()
        self.linear = nn.Linear(embedding_dimension, 1)
        self.act = nn.Sigmoid()

    def forward(self, X, hypergraph):
        # X是从GNN模型中计算出的节点表示
        hyperedge_embedding = hypergraph.v2e(X, aggr="mean")
        score = self.linear(hyperedge_embedding)
        score = self.act(score)
        return score

class HGNNP(nn.Module):
    r"""The HGNN :sup:`+` model proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``out_channels`` (``int``): :math:`C_{out}` is the number of output channels.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNPConv(hid_channels, out_channels, use_bn=use_bn, is_last=True)
        )
        self.pred = ScorePredictor(out_channels)

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)

        return X