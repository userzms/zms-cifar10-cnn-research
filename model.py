"""
CNN模型定义 - 专门为CIFAR10设计
优化版本：添加Mixup数据增强、残差连接、增加模型容量，目标达到93%+准确率
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from config import config


# ============ 修改：CBAM注意力模块 ============
class CBAM(nn.Module):
    """Convolutional Block Attention Module - 通道和空间注意力机制"""

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


# ============ 修改：添加残差连接块 ============
class ResidualBlock(nn.Module):
    """残差块 - 添加跳跃连接以改善梯度流动"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入输出通道数不同或步长不为1，需要使用1x1卷积调整维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # 残差连接
        out = F.relu(out)
        return out


class CIFAR10CNN(pl.LightningModule):
    """为CIFAR10设计的优化CNN模型，目标达到93%+准确率"""

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        # ============ 修改：增加模型容量，通道数从96→192→384→512增加到128→256→512→768 ============
        # 第一阶段卷积
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.cbam1 = CBAM(128)

        # 第二阶段卷积
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.cbam2 = CBAM(256)

        # 第三阶段卷积
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.cbam3 = CBAM(512)

        # 第四阶段卷积
        self.conv4 = nn.Conv2d(512, 768, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(768)
        self.cbam4 = CBAM(768)

        # ============ 修改：在深层添加残差连接以改善训练 ============
        self.res1 = ResidualBlock(512, 512)
        self.res2 = ResidualBlock(768, 768)

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ============ 修改：增加全连接层容量 ============
        self.fc1 = nn.Linear(768, 512)
        self.dropout1 = nn.Dropout(0.5)  # 增加dropout率提升正则化
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, config.num_classes)

        # 损失函数 - 保持标签平滑
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 准确率跟踪
        self.train_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=config.num_classes)

        # ============ 新增：Mixup超参数 ============
        self.mixup_alpha = 0.2  # Mixup混合系数

    def mixup_data(self, x, y):
        """
        实现Mixup数据增强技术
        技术原理：通过线性组合两个样本及其标签创建新的训练样本，提升模型泛化能力
        x: 输入图像
        y: 标签
        """
        if self.mixup_alpha > 0:
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().to(x.device)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Mixup损失函数：混合两个标签的损失"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def forward(self, x):
        # 第一组卷积 + CBAM
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        x = F.max_pool2d(x, 2)

        # 第二组卷积 + CBAM
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x = F.max_pool2d(x, 2)

        # 第三组卷积 + CBAM + 残差连接
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        x = F.max_pool2d(x, 2)
        x = self.res1(x)  # 添加残差连接

        # 第四组卷积 + CBAM + 残差连接
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)
        x = self.res2(x)  # 添加残差连接

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

        # ============ 修改：应用Mixup数据增强 ============
        # 技术原理：Mixup通过混合样本和标签，减少模型对特定样本的过拟合，提升泛化能力
        if torch.rand(1).item() < 0.5:  # 50%概率使用Mixup
            mixed_x, y_a, y_b, lam = self.mixup_data(x, y)
            logits = self(mixed_x)
            loss = self.mixup_criterion(self.criterion, logits, y_a, y_b, lam)

            # 对于Mixup，准确率计算使用混合标签中概率较高的一个
            preds = torch.argmax(logits, dim=1)
            self.train_acc(preds, y_a if lam > 0.5 else y_b)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            self.train_acc(preds, y)

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

    def configure_optimizers(self) -> dict:
        # ============ 修改：GPU优化的学习率调度策略 ============
        # 技术原理：更大的batch size需要相应调整学习率，通常遵循线性缩放原则
        # batch size从64增加到128（2倍），学习率相应调整
        # CosineAnnealingWarmRestarts通过周期性重启学习率，帮助模型跳出局部最优
        optimizer = AdamW(
            self.parameters(),
            lr=0.002,  # 修改：从0.001增加到0.002，配合batch size增加（线性缩放：0.001 * 128/64）
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            amsgrad=True  # 新增：使用AMSGrad变体，提升GPU训练稳定性
        )

        # 使用CosineAnnealingWarmRestarts优化GPU训练
        # 调整参数以适应150 epochs的训练
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=40,  # 修改：从30增加到40，第一个重启周期40个epoch，适应更长训练
            T_mult=2,  # 周期倍增因子
            eta_min=1e-6  # 最小学习率
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # 按epoch更新学习率
                'frequency': 1
            }
        }

    def on_train_epoch_end(self):
        # 每个epoch结束时输出当前学习率
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True)
