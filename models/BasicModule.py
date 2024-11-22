import time
from pathlib import Path

import torch


class BasicModule(torch.nn.Module):
    """
    封装了nn.Module，主要提供save和load两个方法
    """

    def __init__(self):
        super().__init__()
        self.model_name = str(type(self))  # 模型的默认名字

    def load(self, path: str | Path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(torch.load(path, weights_only=False))

    def save(self, name: str | None = None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名，
        如AlexNet_0710_23:57:29.pth
        """
        if name is None:
            prefix = "checkpoints/" + self.model_name + "_"
            name = time.strftime(prefix + "%m%d_%H:%M:%S.pth")
        torch.save(self.state_dict(), name)
        return name

    # def train_step(self, state, data) -> float:
    #     pass
