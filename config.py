"""
配置参数文件
"""
import torch


class Config:
    # 数据参数
    data_dir = "./data"
    batch_size = 64
    num_workers = 0

    # 模型参数
    num_classes = 10
    learning_rate = 0.001
    weight_decay = 1e-4

    # 训练参数
    max_epochs = 30
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