"""
CIFAR10数据加载模块 - GPU优化版本
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import config


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        # 定义数据增强 - GPU优化增强策略
        # 技术原理：更多样化的数据增强结合Mixup能显著提升模型泛化能力
        if config.train_transform:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))  # 随机擦除
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

    def prepare_data(self):
        # 下载数据
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # 加载数据集
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                transform=self.train_transform
            )
            self.val_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                transform=self.test_transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                transform=self.test_transform
            )

    # ============ 修改：优化DataLoader配置以充分利用GPU ============
    # 技术原理：persistent_workers保持worker进程存活，减少epoch间worker初始化开销
    # prefetch_factor控制每个worker预取的batch数量，加速数据传输到GPU
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # 加速CPU到GPU的数据传输
            persistent_workers=True,  # 新增：保持worker进程存活，减少初始化开销
            prefetch_factor=2  # 新增：每个worker预取2个batch，加速数据加载
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,  # 新增：保持worker进程存活
            prefetch_factor=2  # 新增：每个worker预取2个batch
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,  # 新增：保持worker进程存活
            prefetch_factor=2  # 新增：每个worker预取2个batch
        )
