import torch
import torch.nn as nn

import dhg
from dhg.nn import HGNNPConv
from dhg.nn import UniGCNConv, UniGATConv, UniSAGEConv, UniGINConv, MultiHeadWrapper

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

class UniGCN(nn.Module):
    r"""The UniGCN model proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``out_channels`` (``int``): :math:`C_{out}` is the number of output channels.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self, in_channels: int, hid_channels: int, out_channels: int, use_bn: bool = False, drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(UniGCNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        self.layers.append(UniGCNConv(hid_channels, out_channels, use_bn=use_bn, is_last=True))

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X


class UniGAT(nn.Module):
    r"""The UniGAT model proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``out_channels`` (``int``): :math:`C_{out}` is the number of output channels.
        ``num_heads`` (``int``): The Number of attention head in each layer.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): The dropout probability. Defaults to ``0.5``.
        ``atten_neg_slope`` (``float``): Hyper-parameter of the ``LeakyReLU`` activation of edge attention. Defaults to 0.2.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        num_heads: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        atten_neg_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.drop_layer = nn.Dropout(drop_rate)
        self.multi_head_layer = MultiHeadWrapper(
            num_heads,
            "concat",
            UniGATConv,
            in_channels=in_channels,
            out_channels=hid_channels,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
        )
        # The original implementation has applied activation layer after the final layer.
        # Thus, we donot set ``is_last`` to ``True``.
        self.out_layer = UniGATConv(
            hid_channels * num_heads,
            out_channels,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
            is_last=False,
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        X = self.drop_layer(X)
        X = self.multi_head_layer(X=X, hg=hg)
        X = self.drop_layer(X)
        X = self.out_layer(X, hg)
        return X


class UniSAGE(nn.Module):
    r"""The UniSAGE model proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``out_channels`` (``int``): :math:`C_{out}` is the number of output channels.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self, in_channels: int, hid_channels: int, out_channels: int, use_bn: bool = False, drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(UniSAGEConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        self.layers.append(UniSAGEConv(hid_channels, out_channels, use_bn=use_bn, is_last=True))

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X


class UniGIN(nn.Module):
    r"""The UniGIN model proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``out_channels`` (``int``): :math:`C_{out}` is the number of output channels.
        ``eps`` (``float``): The epsilon value. Defaults to ``0.0``.
        ``train_eps`` (``bool``): If set to ``True``, the epsilon value will be trainable. Defaults to ``False``.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        eps: float = 0.0,
        train_eps: bool = False,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            UniGINConv(in_channels, hid_channels, eps=eps, train_eps=train_eps, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            UniGINConv(hid_channels, out_channels, eps=eps, train_eps=train_eps, use_bn=use_bn, is_last=True)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X
