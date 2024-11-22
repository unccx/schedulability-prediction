import torch


def diag2(x: torch.Tensor):
    x[torch.isinf(x)] = 0
    diag_matrix = x.unsqueeze(1) * torch.eye(len(x)).to(x.device)
    return diag_matrix


class V2EMsgPass(torch.nn.Module):

    def __init__(self, aggr: str = "mean", drop_rate: float = 0.0):
        super().__init__()
        self.aggr = aggr
        # self.dropout_layer = torch.nn.Dropout(p=drop_rate)

    def forward(self, X: torch.Tensor, H: torch.Tensor):
        # P = self.dropout_layer(H.T)
        P = H.T
        if self.aggr == "mean":
            X = torch.mm(P, X)
            # D_e_neg_1 = H.sum(dim=0).reciprocal().diag()
            D_e_neg_1 = diag2(H.sum(dim=0).reciprocal())
            X = torch.mm(D_e_neg_1, X)
        elif self.aggr == "sum":
            X = torch.mm(P, X)
        elif self.aggr == "softmax_then_sum":
            P = torch.softmax(P, dim=1)
            X = torch.mm(P, X)
        else:
            raise ValueError(f"Unknown aggregation method {self.aggr}.")

        return X


class E2VMsgPass(torch.nn.Module):

    def __init__(self, aggr: str = "mean", drop_rate: float = 0.0):
        super().__init__()
        self.aggr = aggr
        # self.dropout_layer = torch.nn.Dropout(p=drop_rate)

    def forward(self, X: torch.Tensor, H: torch.Tensor):
        # P = self.dropout_layer(H)
        P = H
        if self.aggr == "mean":
            X = torch.mm(P, X)
            # D_v_neg_1 = H.sum(dim=1).reciprocal().diag()
            D_v_neg_1 = diag2(H.sum(dim=1).reciprocal())
            X = torch.mm(D_v_neg_1, X)
        elif self.aggr == "sum":
            X = torch.mm(P, X)
        elif self.aggr == "softmax_then_sum":
            P = torch.softmax(P, dim=1)
            X = torch.mm(P, X)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggr}")

        return X


class HGNNPConv(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        aggr: str = "mean",
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.theta = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.v2e_msg_pass = V2EMsgPass(aggr, drop_rate)
        self.e2v_msg_pass = E2VMsgPass(aggr, drop_rate)
        self.bn = torch.nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = torch.nn.ReLU(inplace=True)
        self.drop = torch.nn.Dropout(drop_rate)

    def forward(self, X: torch.Tensor, H: torch.Tensor):
        assert torch.all(torch.isfinite(X)), "Tensor contains NaN or Inf!"
        X = self.theta(X)
        assert torch.all(torch.isfinite(X)), "Tensor contains NaN or Inf!"
        X = self.v2e_msg_pass(X, H)
        assert torch.all(torch.isfinite(X)), "Tensor contains NaN or Inf!"
        X = self.e2v_msg_pass(X, H)
        assert torch.all(torch.isfinite(X)), "Tensor contains NaN or Inf!"
        if not self.is_last:
            if self.bn is not None:
                assert torch.all(torch.isfinite(X)), "Tensor contains NaN or Inf!"
                X = self.bn(X)
                assert torch.all(torch.isfinite(X)), "Tensor contains NaN or Inf!"
            X = self.act(X)
            X = self.drop(X)
        return X
