# 配置参数
import torch
class Config:
    # GPU优化的数据参数
    # 技术原理：GPU可以处理更大的batch size，提升训练效率和稳定性
    data_dir = "./data"
    batch_size = 128
    num_workers = 8   # 充分利用GPU的多线程能力，加速数据加载

    # 模型参数
    num_classes = 10
    learning_rate = 0.1  # 配合SGD优化器使用更大的初始学习率
    weight_decay = 5e-4  # 增加到5e-4，配合SGD使用更强的L2正则化，提升权重衰减值


    # 技术原理：SGD+MultiStepLR配合200 epochs的训练，在关键epoch（100, 150, 180）降低学习率
    max_epochs = 200
    accelerator = "gpu"  # GPU训练
    devices = "auto"  # 自动检测GPU数量

    # 数据增强
    train_transform = True

    # 日志和检查点
    log_dir = "./logs"
    checkpoint_dir = "./checkpoints"
    checkpoint_monitor = "val_accuracy"
    checkpoint_mode = "max"

    precision = 32

config = Config()
