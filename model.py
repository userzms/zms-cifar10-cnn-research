"""
CNN模型定义 - 专门为CIFAR10设计
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR  # 修改：使用OneCycleLR替代CosineAnnealingLR
from config import config


# ============ 新增：CBAM注意力模块 ============
class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()

        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # 空间注意力
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        # 通道注意力
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_attention.expand_as(x)

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.sigmoid(self.conv(spatial))

        return x * spatial_attention


class CIFAR10CNN(pl.LightningModule):
    """为CIFAR10设计的CNN模型，目标达到93%+准确率"""

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        # 卷积层 - 保持原有通道数不变
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(96)
        self.cbam1 = CBAM(96)  # 新增：在第一个卷积后添加CBAM

        self.conv2 = nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(192)
        self.cbam2 = CBAM(192)  # 新增：在第二个卷积后添加CBAM

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(384)
        self.cbam3 = CBAM(384)  # 新增：在第三个卷积后添加CBAM

        self.conv4 = nn.Conv2d(384, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.cbam4 = CBAM(512)  # 新增：在第四个卷积后添加CBAM

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 全连接层 - 微调全连接层结构
        self.fc1 = nn.Linear(512, 384)
        self.dropout1 = nn.Dropout(0.4)  # 修改：增加dropout率从0.3到0.4
        self.fc2 = nn.Linear(384, 192)
        self.dropout2 = nn.Dropout(0.3)  # 修改：增加dropout率从0.2到0.3
        self.fc3 = nn.Linear(192, config.num_classes)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 新增：使用标签平滑

        # 准确率跟踪
        self.train_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=config.num_classes)

    def forward(self, x):
        # 第一组卷积 + CBAM
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)  # 新增：应用CBAM注意力
        x = F.max_pool2d(x, 2)

        # 第二组卷积 + CBAM
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)  # 新增：应用CBAM注意力
        x = F.max_pool2d(x, 2)

        # 第三组卷积 + CBAM
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)  # 新增：应用CBAM注意力
        x = F.max_pool2d(x, 2)

        # 第四组卷积 + CBAM
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)  # 新增：应用CBAM注意力

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
        # 修改：使用更高的初始学习率和OneCycleLR策略
        optimizer = AdamW(
            self.parameters(),
            lr=0.01,  # 修改：提高学习率从0.001到0.01
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)  # 新增：明确指定AdamW的betas参数
        )

        # 修改：使用OneCycleLR替代CosineAnnealingLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.01,  # 最大学习率
            epochs=config.max_epochs,
            steps_per_epoch=782,  # 782 batches per epoch (50000/64 ≈ 782)
            pct_start=0.3,  # 前30%的步骤用于学习率上升
            anneal_strategy='cos',  # 余弦退火
            div_factor=25.0,  # 初始学习率 = max_lr / div_factor
            final_div_factor=10000.0  # 最终学习率 = initial_lr / final_div_factor
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # 修改：改为每个step更新学习率
                'frequency': 1
            }
        }

    def on_train_epoch_end(self):
        # 每个epoch结束时输出当前学习率
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True)