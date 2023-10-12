import torch
from torch.utils.data import Dataset
from dhg import Hypergraph
import csv
from pathlib import Path
import random
from typing import Union

class LinkPredictDataset(Dataset):
    def __init__(
            self, dataset_name:Union[str, Path],
            root_dir:Path=Path("./../EDF/data"), 
            data_balance: bool=True, 
            ratio:tuple[float, float]=(0.7, 0.3),
            device:torch.device=None
            ):
        self.data_path:Path = root_dir / dataset_name
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.hyperedge_list = self.read_hyperedge_list_from_csv(self.data_path / "hyperedges.csv")
        self.neg_hyperedge_list = self.read_hyperedge_list_from_csv(self.data_path / "negative_samples.csv")
        self.features:torch.Tensor = self.read_features_from_csv(self.data_path / "task_quadruples.csv").to(self.device)
        self.processor_speeds:list[float] = self.read_processor_speed_from_csv(self.data_path / "platform.csv")
        self.msg_pass_hyperedge_list, self.pos_hyperedge_list = self.split_hyperedge_list(self.hyperedge_list, ratio=ratio)
        self.data_balance:bool = data_balance

        self.cache = {}

        # 处理监督边和负采样边数据不平衡的问题
        if self.data_balance:
            length = min(len(self.neg_hyperedge_list), len(self.pos_hyperedge_list))
            self.neg_hyperedge_list = random.sample(self.neg_hyperedge_list, length)
            self.pos_hyperedge_list = random.sample(self.pos_hyperedge_list, length)

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
    def original_hg(self):
        if self.cache.get("original_hg", None) is None:
            self.cache["original_hg"] = Hypergraph(self.num_v, self.hyperedge_list, device=self.device)
        return self.cache["original_hg"]
    
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
        fastest_processor_speed = self.processor_speeds[0]
        platform_speed = [processor_speed / fastest_processor_speed for processor_speed in self.processor_speeds] # 处理器速度归一化
        sum_of_normalized_speed = sum(platform_speed) # S_m
        return sum_of_normalized_speed
    
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
    def normalized_utilization(self):
        """normalized_utilization的shape是(self.num_v, 1)"""
        if self.cache.get("normalized_utilization", None) is None:
            self.cache["normalized_utilization"] = self.X.index_select(dim=1, index=torch.tensor(3).to(self.device))
        return self.cache["normalized_utilization"]
    
    @property
    def pos_hg_system_utilization_distribution(self):
        """pos_hg_system_utilization_distribution的shape是(num_pos_e,)"""
        if self.cache.get("pos_hg_system_utilization_distribution", None) is None:
            sum_of_normalized_utilization_pos = torch.sparse.mm(self.pos_hg.H_T, self.normalized_utilization).squeeze()
            self.cache["pos_hg_system_utilization_distribution"] = sum_of_normalized_utilization_pos / self.S_m
        return self.cache["pos_hg_system_utilization_distribution"]

    @property
    def neg_hg_system_utilization_distribution(self):
        """neg_hg_system_utilization_distribution的shape是(num_neg_e,)"""
        if self.cache.get("neg_hg_system_utilization_distribution", None) is None:
            sum_of_normalized_utilization_neg = torch.sparse.mm(self.pos_hg.H_T, self.normalized_utilization).squeeze()
            self.cache["neg_hg_system_utilization_distribution"] = sum_of_normalized_utilization_neg / self.S_m
        return self.cache["neg_hg_system_utilization_distribution"]
    
    @property
    def all_system_utilization_distribution(self):
        """all_system_utilization_distribution的shape是(num_pos_e + num_neg_e,)"""
        all_system_utilization = torch.cat((self.pos_hg_system_utilization_distribution, self.neg_hg_system_utilization_distribution), dim=0)
        return all_system_utilization

    @staticmethod

    @staticmethod
    def read_hyperedge_list_from_csv(file_path: Path):
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            hyperedge_list = [[int(v) for v in row] for row in reader]
        return hyperedge_list
    
    @staticmethod
    def read_features_from_csv(file_path: Path):
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            features = [[float(feature) for feature in row] for row in reader]
        return torch.tensor(features)
    
    @staticmethod
    def read_processor_speed_from_csv(file_path: Path):
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            processor_speeds = [float(speed) for _, speed in reader]
        return processor_speeds
    
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