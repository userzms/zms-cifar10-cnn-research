"""
配置参数文件
"""
import torch


class Config:
    # 数据参数
    data_dir = "./data"
    batch_size = 64
    num_workers = 4

    # 模型参数
    num_classes = 10
    learning_rate = 0.01
    weight_decay = 1e-4

    # 训练参数
    max_epochs = 120  # 修改：增加训练轮数从100到120，配合warm restart策略获得更好收敛
    accelerator = "cpu"
    devices = 1

    # 数据增强
    train_transform = True  # 是否使用数据增强

    # 日志和检查点
    log_dir = "./logs"
    checkpoint_dir = "./checkpoints"
    checkpoint_monitor = "val_accuracy"
    checkpoint_mode = "max"

    precision = 32

config = Config()