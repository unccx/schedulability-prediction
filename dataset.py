import random
from math import ceil, log
from pathlib import Path
from typing import Optional, Union

import torch
from dhg import Hypergraph
from simrt.core import TaskInfo
from simrt.core.processor import PlatformInfo
from simrt.utils import TaskStorage
from simrt.utils.task_storage import TaskStorage
from torch.utils.data import Dataset


class LinkPredictDataset(Dataset):

    def __init__(
        self,
        dataset_path: str | Path,
        msgpass_ratio: float,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.task_db = TaskStorage(Path(dataset_path) / "data.sqlite")
        metadata = self.task_db.get_metadata()
        self.platform = PlatformInfo(metadata["speed_list"])

        # 实际可调度映射为可调度超边
        self.true_hyperedge_list: list[list[int]] = (
            self.read_hyperedge_list_from_sqlite(is_schedulable=True)
        )
        # 实际不可调度映射为不可调度超边
        self.false_hyperedge_list: list[list[int]] = (
            self.read_hyperedge_list_from_sqlite(is_schedulable=False)
        )
        # 节点特征
        self.features: torch.Tensor = self.read_features_from_sqlite().to(self.device)

        # 将可调度超边划分为消息传递边和用于监督学习的正样本边
        self.msgpass_hyperedge_list, self.pos_hyperedge_list = (
            self.split_hyperedge_list(
                self.true_hyperedge_list,
                msgpass_sample_ratio=(msgpass_ratio, 1 - msgpass_ratio),
            )
        )
        # 用于监督学习的负样本边
        self.neg_hyperedge_list: list[list[int]] = self.negative_sampling(
            len(self.pos_hyperedge_list)
        )

        self.hg = Hypergraph(num_v=self.num_v, device=self.device)
        self.hg.add_hyperedges(e_list=self.msgpass_hyperedge_list, group_name="msgpass")
        self.hg.add_hyperedges(e_list=self.pos_hyperedge_list, group_name="positive")
        self.hg.add_hyperedges(e_list=self.neg_hyperedge_list, group_name="negative")

        self.msgpass_hg = Hypergraph(
            num_v=self.num_v, e_list=self.msgpass_hyperedge_list, device=self.device
        )
        self.pos_hg = Hypergraph(
            num_v=self.num_v, e_list=self.pos_hyperedge_list, device=self.device
        )
        self.neg_hg = Hypergraph(
            num_v=self.num_v, e_list=self.neg_hyperedge_list, device=self.device
        )

    def negative_sampling(self, num_sample: int):
        """从不可调度超边中选出负样本边"""
        if num_sample > len(self.false_hyperedge_list):
            raise ValueError("num_sample <= len(self.false_hyperedge_list)")
        return random.sample(self.false_hyperedge_list, num_sample)

    @property
    def num_v(self):
        """节点数量"""
        return self.features.shape[0]

    @property
    def feat_dim(self):
        """节点特征维度"""
        return self.features.shape[1]

    @property
    def X(self):
        "节点特征矩阵"
        # return torch.eye(self.num_v) # 不使用任务属性作为节点嵌入向量的初始化。使用one-hot向量
        return self.features

    @property
    def S_m(self):
        return self.platform.S_m

    @property
    def utilizations(self):
        """utilization的shape是(self.num_v, 1)"""
        return self.X.index_select(dim=1, index=torch.tensor(3).to(self.device))

    @property
    def sched_hg_hyperedge_size(self):
        return torch.Tensor(
            [len(hyperedge) for hyperedge in self.true_hyperedge_list]
        ).to(self.device)

    @property
    def pos_hg_hyperedge_size(self):
        return torch.Tensor(
            [len(hyperedge) for hyperedge in self.pos_hyperedge_list]
        ).to(self.device)

    @property
    def neg_hg_hyperedge_size(self):
        return torch.Tensor(
            [len(hyperedge) for hyperedge in self.neg_hyperedge_list]
        ).to(self.device)

    @property
    def sched_hg_system_utilizations(self):
        sched_hg = self.hg.clone()
        sched_hg.remove_group("negative")
        utilizations = torch.sparse.mm(sched_hg.H_T, self.utilizations).squeeze()
        return utilizations / self.S_m

    @property
    def pos_hg_system_utilizations(self):
        utilizations = torch.sparse.mm(
            self.hg.H_T_of_group("positive"), self.utilizations
        ).squeeze()
        return utilizations / self.S_m

    @property
    def neg_hg_system_utilizations(self):
        utilizations = torch.sparse.mm(
            self.hg.H_T_of_group("negative"), self.utilizations
        ).squeeze()
        return utilizations / self.S_m

    def read_hyperedge_list_from_sqlite(self, is_schedulable: bool) -> list[list[int]]:
        values = self.task_db.get_tasksets_dict(
            is_schedulable=is_schedulable, show_progress=True
        ).values()

        hyperedge_list = [[taskinfo.id for taskinfo in value[0]] for value in values]
        return hyperedge_list

    def read_features_from_sqlite(self):
        taskinfos = self.task_db.get_all_taskinfos()

        features = [
            [
                taskinfo.wcet,
                taskinfo.deadline,
                taskinfo.period,
                taskinfo.utilization,
                log((1 + taskinfo.wcet)),
                log((1 + taskinfo.deadline)),
                log((1 + taskinfo.period)),
                (taskinfo.period - taskinfo.wcet),
                (taskinfo.period - taskinfo.deadline),
                (taskinfo.deadline - taskinfo.wcet),
            ]
            for taskinfo in taskinfos
        ]
        return torch.tensor(features)

    @staticmethod
    def split_hyperedge_list(
        hyperedge_list: list[list[int]], msgpass_sample_ratio: tuple[float, float]
    ):
        assert (
            msgpass_sample_ratio[0] + msgpass_sample_ratio[1] == 1.0
        ), f"The sum of ratio is not equal to 1."

        # 将超图的边分为消息传递边和监督边
        shuffled_hyperedge_list: list[list[int]] = hyperedge_list.copy()
        random.shuffle(shuffled_hyperedge_list)
        # 获取前ratio比例的元素作为消息传递边的列表
        split_pos = int(len(shuffled_hyperedge_list) * msgpass_sample_ratio[0])
        # 获取前ratio比例的元素作为消息传递边的列表
        msg_pass_hyperedge_list = shuffled_hyperedge_list[:split_pos]
        # 获取后ratio比例的元素作为监督边的列表
        pos_hyperedge_list = shuffled_hyperedge_list[split_pos:]
        return msg_pass_hyperedge_list, pos_hyperedge_list

    def __len__(self):
        return len(self.pos_hyperedge_list)

    def __getitem__(self, index):
        return self.pos_hyperedge_list[index], self.neg_hyperedge_list[index]
