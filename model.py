import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from config import config


# CBAM注意力模块
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


# 残差连接块
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
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        # 进一步增加模型容量
        # 技术原理：更大的通道数提供更强的表达能力，配合SGD优化器获得更好的收敛
        # 第一阶段卷积
        self.conv1 = nn.Conv2d(3, 160, kernel_size=3, padding=1, bias=False)  # 从128增加到160
        self.bn1 = nn.BatchNorm2d(160)
        self.cbam1 = CBAM(160)

        # 第二阶段卷积
        self.conv2 = nn.Conv2d(160, 320, kernel_size=3, padding=1, bias=False)  # 从256增加到320
        self.bn2 = nn.BatchNorm2d(320)
        self.cbam2 = CBAM(320)

        # 第三阶段卷积
        self.conv3 = nn.Conv2d(320, 640, kernel_size=3, padding=1, bias=False)  # 从512增加到640
        self.bn3 = nn.BatchNorm2d(640)
        self.cbam3 = CBAM(640)

        # 第四阶段卷积
        self.conv4 = nn.Conv2d(640, 960, kernel_size=3, padding=1, bias=False)  # 从768增加到960
        self.bn4 = nn.BatchNorm2d(960)
        self.cbam4 = CBAM(960)

        # 残差连接
        self.res1 = ResidualBlock(640, 640)
        self.res2 = ResidualBlock(640, 640)
        self.res3 = ResidualBlock(960, 960)
        self.res4 = ResidualBlock(960, 960)

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 增加全连接层容量
        self.fc1 = nn.Linear(960, 640)  # 从768->512 改为 960->640
        self.dropout1 = nn.Dropout(0.4)  # 从0.5降低到0.4
        self.fc2 = nn.Linear(640, 512)  # 从512->384 改为 640->512
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 320)  # 从384->256 改为 512->320
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(320, config.num_classes)


        # 使用0.1的标签平滑，配合更强的正则化（SGD+Momentum）
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 准确率跟踪
        self.train_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=config.num_classes)


        self.use_mixup = False
        self.mixup_alpha = 0.2
        self.mixup_prob = 0.05

    def mixup_data(self, x, y):
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
        """Mixup损失函数"""
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
        x = self.res1(x)
        x = self.res2(x)

        # 第四组卷积 + CBAM + 残差连接
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)
        x = self.res3(x)
        x = self.res4(x)

        # 全局平均池化
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.use_mixup and torch.rand(1).item() < self.mixup_prob:
            mixed_x, y_a, y_b, lam = self.mixup_data(x, y)
            logits = self(mixed_x)
            loss = self.mixup_criterion(self.criterion, logits, y_a, y_b, lam)

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
        # 使用SGD+Momentum优化器
        # 技术原理：SGD+Momentum在CIFAR10上通常比AdamW表现更好，能获得更好的泛化性能
        optimizer = SGD(
            self.parameters(),
            lr=0.1,  # 从0.001增加到0.1，SGD需要更大的初始学习率
            momentum=0.9,  # SGD的标准momentum值
            weight_decay=5e-4,  # 从1e-4增加到5e-4，配合SGD使用更强的L2正则化
            nesterov=True  # 使用Nesterov momentum，能获得更好的收敛效果
        )

        # 使用MultiStepLR学习率调度
        # 技术原理：MultiStepLR在CIFAR10上被广泛使用，在特定epoch降低学习率
        # 是ImageNet和CIFAR10的经典学习率策略
        scheduler = MultiStepLR(
            optimizer,
            milestones=[100, 150, 180],  # 在第100、150、180个epoch降低学习率
            gamma=0.1  # 每次学习率降低为原来的1/10
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
