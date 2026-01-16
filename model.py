"""
CNN模型定义 - 专门为CIFAR10设计
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy # 添加
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import config


class CIFAR10CNN(pl.LightningModule):
    """为CIFAR10设计的CNN模型，目标达到93%+准确率"""

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        # 卷积层
        # 增加通道数
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1, bias=False)  # 64→96
        self.bn1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False)  # 128→192
        self.bn2 = nn.BatchNorm2d(192)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False)  # 256→384
        self.bn3 = nn.BatchNorm2d(384)

        self.conv4 = nn.Conv2d(384, 512, kernel_size=3, padding=1, bias=False)  # 保持512
        self.bn4 = nn.BatchNorm2d(512)

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 全连接层
        # 增加全连接层容量
        self.fc1 = nn.Linear(512, 384)  # 256→384
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(384, 192)  # 128→192
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(192, config.num_classes)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 准确率跟踪
        self.train_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=config.num_classes)

    def forward(self, x):
        # 第一组卷积
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        # 第二组卷积
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        # 第三组卷积
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # 第四组卷积
        x = F.relu(self.bn4(self.conv4(x)))

        # 全局平均池化
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)

        # 记录日志
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', self.val_acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)

        self.log('test_loss', loss)
        self.log('test_accuracy', self.test_acc)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.max_epochs,
            eta_min=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def on_train_epoch_end(self):
        # 每个epoch结束时输出当前学习率
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True)