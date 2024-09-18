import csv
import random
from math import ceil, log
from pathlib import Path
from typing import Optional, Union

import torch
from dhg import Hypergraph
from simRT.core import TaskInfo
from simRT.generator import HGConfig
from simRT.utils import TaskStorage
from torch.utils.data import Dataset


class LinkPredictDataset(Dataset):

    def __init__(
        self,
        dataset_name: Union[str, Path],
        root_dir: Path,
        data_balance: bool = True,
        ratio: tuple[float, float] = (0.7, 0.3),
        device: Optional[torch.device] = None,
    ):

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.data_path: Path = root_dir / dataset_name
        self.task_db = TaskStorage(self.data_path / "data.sqlite")
        self.hgconfig = HGConfig.from_json(self.data_path / "config.json")
        self.processor_speeds: list[float] = self.hgconfig.platform_info.speed_list

        self.hyperedge_list = self.read_hyperedge_list_from_sqlite(is_schedulable=True)
        self.neg_hyperedge_list = self.read_hyperedge_list_from_sqlite(
            is_schedulable=False
        )
        self.features: torch.Tensor = self.read_features_from_sqlite().to(self.device)

        self.msg_pass_hyperedge_list, self.pos_hyperedge_list = self.split_hyperedge_list(self.hyperedge_list, ratio=ratio)
        self.data_balance:bool = data_balance

        self.cache = {}

        # 处理监督边和负采样边数据不平衡的问题
        if self.data_balance:
            length = min(len(self.neg_hyperedge_list), len(self.pos_hyperedge_list))
            self.neg_hyperedge_list = random.sample(self.neg_hyperedge_list, length)
            self.pos_hyperedge_list = random.sample(self.pos_hyperedge_list, length)
        else:
            length = ceil(len(self.neg_hyperedge_list) * ratio[1])
            self.neg_hyperedge_list = random.sample(self.neg_hyperedge_list, length)

    @property
    def num_v(self):
        return self.features.shape[0]

    @property
    def feat_dim(self):
        return self.features.shape[1]

    @property
    def num_pos_e(self):
        return len(self.pos_hyperedge_list)

    @property
    def num_neg_e(self):
        return len(self.neg_hyperedge_list)

    @property
    def X(self):
        # return torch.eye(self.num_v) # 不使用任务属性作为节点嵌入向量的初始化。使用one-hot向量
        return self.features

    @property
    def schedulable_hg(self):
        """schedulable_hg表示所有可调度任务集包括作为数据的正例和消息传递边"""
        if self.cache.get("schedulable_hg", None) is None:
            self.cache["schedulable_hg"] = Hypergraph(
                self.num_v, self.hyperedge_list, device=self.device
            )
        return self.cache["schedulable_hg"]

    @property
    def msg_pass_hg(self):
        if self.cache.get("msg_pass_hg", None) is None:
            self.cache["msg_pass_hg"] = Hypergraph(self.num_v, self.msg_pass_hyperedge_list, device=self.device)
        return self.cache["msg_pass_hg"]

    @property
    def pos_hg(self):
        if self.cache.get("pos_hg", None) is None:
            self.cache["pos_hg"] = Hypergraph(self.num_v, self.pos_hyperedge_list, device=self.device)
        return self.cache["pos_hg"]

    @property
    def neg_hg(self):
        if self.cache.get("neg_hg", None) is None:
            self.cache["neg_hg"] = Hypergraph(self.num_v, self.neg_hyperedge_list, device=self.device)
        return self.cache["neg_hg"]

    @property
    def S_m(self):
        return self.hgconfig.platform_info.S_m

    @property
    def data(self):
        msg_pass_data = {"hyperedge_list": self.msg_pass_hyperedge_list, "num_edges" : len(self.msg_pass_hyperedge_list)}
        pos_data      = {"hyperedge_list": self.pos_hyperedge_list, "num_edges" : len(self.pos_hyperedge_list)}
        neg_data      = {"hyperedge_list": self.neg_hyperedge_list, "num_edges" : len(self.neg_hyperedge_list)}

        return {
            "msg_pass": msg_pass_data,
            "pos": pos_data,
            "neg": neg_data,
            "vertices_feature": self.features,
            "num_vertices": self.num_v,
            "platform": self.processor_speeds
        }

    @property
    def utilizations(self):
        """utilization的shape是(self.num_v, 1)"""
        if self.cache.get("utilization", None) is None:
            self.cache["utilization"] = self.X.index_select(
                dim=1, index=torch.tensor(3).to(self.device)
            )
        return self.cache["utilization"]

    @property
    def schedulable_hg_hyperedge_size(self):
        if self.cache.get("schedulable_hg_hyperedge_size", None) is None:

            self.cache["schedulable_hg_hyperedge_size"] = torch.Tensor(
                [len(e) for e in self.schedulable_hg.e[0]]
            ).to(self.device)
        return self.cache["schedulable_hg_hyperedge_size"]

    @property
    def pos_hg_hyperedge_size(self):
        if self.cache.get("pos_hg_hyperedge_size", None) is None:

            self.cache["pos_hg_hyperedge_size"] = torch.Tensor(
                [len(e) for e in self.pos_hg.e[0]]
            ).to(self.device)
        return self.cache["pos_hg_hyperedge_size"]

    @property
    def neg_hg_hyperedge_size(self):
        if self.cache.get("neg_hg_hyperedge_size", None) is None:

            self.cache["neg_hg_hyperedge_size"] = torch.Tensor(
                [len(e) for e in self.neg_hg.e[0]]
            ).to(self.device)
        return self.cache["neg_hg_hyperedge_size"]

    @property
    def schedulable_hg_system_utilizations(self):
        """schedulable_hg表示所有可调度任务集包括作为数据的正例和消息传递边"""
        if (
            self.cache.get("schedulable_hg_system_utilization_distribution", None)
            is None
        ):
            original_taskset_utilizations = torch.sparse.mm(
                self.schedulable_hg.H_T, self.utilizations
            ).squeeze()
            self.cache["schedulable_hg_system_utilization_distribution"] = (
                original_taskset_utilizations / self.S_m
            )
        return self.cache["schedulable_hg_system_utilization_distribution"]

    @property
    def pos_hg_system_utilizations(self):
        """pos_hg_system_utilization_distribution的shape是(num_pos_e,)"""
        if self.cache.get("pos_hg_system_utilization_distribution", None) is None:
            pos_taskset_utilizations = torch.sparse.mm(
                self.pos_hg.H_T, self.utilizations
            ).squeeze()
            self.cache["pos_hg_system_utilization_distribution"] = (
                pos_taskset_utilizations / self.S_m
            )
        return self.cache["pos_hg_system_utilization_distribution"]

    @property
    def neg_hg_system_utilizations(self):
        """neg_hg_system_utilization_distribution的shape是(num_neg_e,)"""
        if self.cache.get("neg_hg_system_utilization_distribution", None) is None:
            neg_taskset_utilizations = torch.sparse.mm(
                self.neg_hg.H_T, self.utilizations
            ).squeeze()
            self.cache["neg_hg_system_utilization_distribution"] = (
                neg_taskset_utilizations / self.S_m
            )
        return self.cache["neg_hg_system_utilization_distribution"]

    def read_hyperedge_list_from_sqlite(self, is_schedulable: bool):
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
    def split_hyperedge_list(hyperedge_list:list[list[int]], ratio:tuple[float, float]=(0.7, 0.3)):
        assert ratio[0] + ratio[1] == 1.0, f"The sum of ratio is not equal to 1."

        # 将超图的边分为消息传递边和监督边
        shuffled_hyperedge_list:list[list[int]] = hyperedge_list.copy()
        random.shuffle(shuffled_hyperedge_list)
        split_pos = int(len(shuffled_hyperedge_list) * ratio[0]) # 获取前ratio比例的元素作为消息传递边的列表
        msg_pass_hyperedge_list = shuffled_hyperedge_list[:split_pos] # 获取前ratio比例的元素作为消息传递边的列表
        pos_hyperedge_list = shuffled_hyperedge_list[split_pos:] # 获取后ratio比例的元素作为监督边的列表
        return msg_pass_hyperedge_list, pos_hyperedge_list

    def __len__(self):
        if self.data_balance:
            return min(len(self.neg_hyperedge_list), len(self.pos_hyperedge_list))
        return len(self.pos_hyperedge_list)

    def __getitem__(self, idx):
        return self.pos_hyperedge_list[idx], self.neg_hyperedge_list[idx]
