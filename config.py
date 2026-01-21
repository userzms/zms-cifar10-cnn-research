"""
配置参数文件
GPU优化版本：针对GPU训练优化参数，目标达到93%+准确率
"""
import torch


class Config:
    # ============ 修改：GPU优化的数据参数 ============
    # 技术原理：GPU可以处理更大的batch size，提升训练效率和稳定性
    data_dir = "./data"
    batch_size = 128  # 修改：从64增加到128，GPU可以处理更大的batch，提升训练稳定性
    num_workers = 8   # 修改：从4增加到8，充分利用GPU的多线程能力，加速数据加载

    # 模型参数
    num_classes = 10
    learning_rate = 0.001  # 修改：从0.01降低到0.001，配合更大的batch size使用更小的学习率
    weight_decay = 1e-4

    # ============ 修改：进一步增加训练轮数以达成93%目标 ============
    # 技术原理：从150增加到200 epochs，配合更深的网络和更复杂的优化策略，给模型更多时间充分学习和收敛
    max_epochs = 200  # 修改：从150增加到200，给模型更多时间充分学习和收敛
    accelerator = "gpu"  # GPU训练
    devices = "auto"  # 自动检测GPU数量

    # 数据增强
    train_transform = True  # 是否使用数据增强

    # 日志和检查点
    log_dir = "./logs"
    checkpoint_dir = "./checkpoints"
    checkpoint_monitor = "val_accuracy"
    checkpoint_mode = "max"

    precision = 32  # 保持32位浮点精度，如果GPU支持可以改为16使用混合精度训练

config = Config()
