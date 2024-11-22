import warnings

from utils.metric import LinkPredictionEvaluator


class DefaultConfig:
    model = "HGNNPSchedulabilityPredictor"  # 使用的模型，名字必须与models/__init__.py中的名字一致
    hid_channels = 1024
    out_channels = 1024
    use_bn = False

    seed = 2024
    train_data_root = "data/2024-12-01_22-57-57"  # 训练集存放路径
    val_data_root = "data/2024-12-02_11-03-10"
    test_data_root = "data/2024-12-02_20-36-35"  # 测试集存放路径
    load_model_path: str | None = None  # 加载预训练的模型的路径，为 None 代表不加载

    msgpass_ratio = 0.8
    batch_size = 4096  # batch size
    use_gpu = True  # use GPU or not
    num_workers = 8  # how many workers for loading data

    max_epoch = 200
    lr = 0.001  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数


def parse(self, kwargs):
    for key, val in kwargs.items():
        if not hasattr(self, key):
            warnings.warn("Warning: opt has not attribute {key}")
        setattr(self, key, val)

    print("user config:")
    for key, val in self.__class__.__dict__.items():
        if not key.startswith("__"):
            print(key, getattr(self, key))


DefaultConfig.parse = parse  # type: ignore
CONFIG = DefaultConfig()
