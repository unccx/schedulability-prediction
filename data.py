import torch
from torch.utils.data import Dataset
import dhg
from dhg import Hypergraph
import csv
from pathlib import Path
import random
from typing import Union, List, Dict
import itertools

class LinkPredictDataset(Dataset):
    def __init__(
            self, dataset_name:Union[str, Path],
            root_dir:Path=Path("./../EDF/data"), 
            data_balance:bool=True, 
            portion:Dict={"msg_pass":4, "supervision_pos":4, "validation_pos":1, "test_pos":1}, # 这是消息传递边和监督边的比例
            phase:str="train",
            device:torch.device=None,
            seed:int=2023
            ):
        assert phase in ["train", "validate", "test"]
        self.phase = phase
        self.data_path:Path = root_dir / dataset_name

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.features:torch.Tensor = self.read_features_from_csv(self.data_path / "task_quadruples.csv").to(self.device)
        self.processor_speeds:list[float] = self.read_processor_speed_from_csv(self.data_path / "platform.csv")

        # 在Transductive Setting中
        # hyperedge_list需要划分为四部分：训练时消息传递超边、训练时监督正例、验证正例、测试正例
        # negative_sampling需要划分为三部分：               训练时监督负例、验证负例、测试负例
        self.seed = seed
        dhg.random.set_seed(self.seed) # 保证导入同一数据集，分别选择不同phase时的划分结果相同，以及后续数据平衡时随机一致

        self.hyperedge_list = self.read_hyperedge_list_from_csv(self.data_path / "hyperedges.csv")
        negative_sampling = self.read_hyperedge_list_from_csv(self.data_path / "negative_samples.csv")

        # hyperedge_list先划分为两部分：训练时消息传递超边、pos_hyperedge_list（训练时监督正例、验证正例、测试正例）
        self.portion = portion.copy()
        partition_sum =  sum(self.portion.values())
        split_ratios = {key: value / partition_sum for key, value in self.portion.items()}
        split_ratios = {"msg_pass":split_ratios["msg_pass"], "pos":1-split_ratios["msg_pass"]}
        msg_pass_hyperedge_list, pos_hyperedge_list = self.split_hyperedge_list(self.hyperedge_list, ratios=split_ratios)

        # 处理监督边和负采样边数据不平衡的问题
        self.data_balance:bool = data_balance
        if self.data_balance:
            length = min(len(negative_sampling), len(pos_hyperedge_list))
            negative_sampling = random.sample(negative_sampling, length)
            pos_hyperedge_list = random.sample(pos_hyperedge_list, length)

        del self.portion["msg_pass"]
        partition_sum =  sum(self.portion.values())
        split_ratios = {key: value / partition_sum for key, value in self.portion.items()}

        # pos_hyperedge_list在与self.neg_hyperedge_list进行数据平衡后，又会划分为三部分。因此不需要在数据集中保存
        # pos_hyperedge_list划分为三部分：训练时监督正例、验证正例、测试正例
        (self.train_pos_hyperedge_list, 
         self.validate_pos_hyperedge_list, 
         self.test_pos_hyperedge_list) = self.split_hyperedge_list(pos_hyperedge_list, ratios=split_ratios)
        
        # self.neg_hyperedge_list划分为三部分：训练时监督负例、验证负例、测试负例
        (self.train_neg_hyperedge_list, 
         self.validate_neg_hyperedge_list, 
         self.test_neg_hyperedge_list) = self.split_hyperedge_list(negative_sampling, ratios=split_ratios)

        if self.phase == "train":
            self.msg_pass_hyperedge_list = msg_pass_hyperedge_list
            self.pos_hyperedge_list = self.train_pos_hyperedge_list
            self.neg_hyperedge_list = self.train_neg_hyperedge_list
        elif self.phase == "validate":
            self.msg_pass_hyperedge_list = msg_pass_hyperedge_list + self.train_pos_hyperedge_list
            self.pos_hyperedge_list = self.validate_pos_hyperedge_list
            self.neg_hyperedge_list = self.validate_neg_hyperedge_list
        else: # self.phase == "test"
            self.msg_pass_hyperedge_list = msg_pass_hyperedge_list + self.train_pos_hyperedge_list + self.validate_pos_hyperedge_list
            self.pos_hyperedge_list = self.test_pos_hyperedge_list
            self.neg_hyperedge_list = self.test_neg_hyperedge_list

        self.cache = {}

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
    def split_hyperedge_list(hyperedge_list:list[list[int]], ratios:Dict) -> List[List[List[int]]]:
        assert len([v for k, v in ratios.items() if v <= 0 or v >= 1]) == 0
        assert abs(1 - sum(ratios.values())) < 1e-10, f"The sum of ratio is not equal to 1."

        shuffled_hyperedge_list:list[list[int]] = hyperedge_list.copy()
        random.shuffle(shuffled_hyperedge_list)

        # 按照ratios的比例计算出划分列表后的右索引
        ratios = ratios.values()
        right = [int(len(shuffled_hyperedge_list) * accumulate_ratio) for accumulate_ratio in itertools.accumulate(ratios)]
        left = [0]+right[:-1]
        split_lists = []
        for l, r in zip(left, right):
            split_lists.append(shuffled_hyperedge_list[l:r])

        # List[int]是超边的类型，List[List[int]]是超边列表的类型，List[List[List[int]]]是以划分成若干份的超边列表为元素的列表
        return split_lists # 使用序列解包接收划分后的超边列表碎片

    def __len__(self):
        return min(len(self.neg_hyperedge_list), len(self.pos_hyperedge_list))

    def __getitem__(self, idx):
        return self.pos_hyperedge_list[idx], self.neg_hyperedge_list[idx]