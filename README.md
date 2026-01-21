# CIFAR10 CNN Classification - GPU Optimized Version
PyTorch Lightning CNN for CIFAR10 classification, optimized for GPU training to achieve **93%+ test accuracy**.
## Project Goal
使用PyTorch Lightning构建CNN模型，在CIFAR10数据集上达到93%以上的测试准确率（GPU训练版本）。
### 模型结构
- **输入**: 32×32×3 RGB图像
- **4个卷积块**，每个块包含CBAM注意力机制：
  - 块1: 3 → 128 通道
  - 块2: 128 → 256 通道
  - 块3: 256 → 512 通道 + 残差连接
  - 块4: 512 → 768 通道 + 残差连接
- **全局平均池化**
- **3个全连接层**: 768 → 512 → 256 → 10
- **总参数量**: ~4.8M (从~2.9M增加)
### 关键特性
- **残差连接** - 解决深度网络的梯度消失问题
- **CBAM注意力** - 通道和空间注意力机制
- **Mixup数据增强** - 提升泛化能力
- **Warm Restart调度器** - 跳出局部最优
- **标签平滑** - 防止过拟合
- **梯度裁剪** - 提升训练稳定性
### 环境要求
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- GPU服务器访问权限 (SSH)
- 8GB+ GPU显存 (推荐)
- 详见requirements.txt
### 安装依赖
```bash
pip install -r requirements.txt
```
### 训练
**快速测试:**
```bash
python train.py --test
```
**正式训练：**
```bash
python train.py

```
## 项目结构
```
zms_cifar10_cnn/
├── model.py                  # CNN模型架构（CBAM + ResNet块）
├── data_module.py            # 数据加载和增强
├── config.py                 # 配置参数
├── train.py                  # 训练脚本
├── detailed_verify.py        # 数据集验证
├── requirements.txt          # Python依赖
├── checkpoints/              # 保存的模型检查点
├── logs/                     # TensorBoard日志
└── data/                     # CIFAR10数据集（自动下载）
```

## 配置说明
### 关键参数 (config.py)
| 参数 | 值      | 说明 |
|------|--------|------|
| `batch_size` | 128    | 训练批次大小（GPU优化） |
| `num_workers` | 8      | 数据加载线程数 |
| `learning_rate` | 0.001  | 初始学习率 |
| `max_epochs` | 200    | 最大训练轮数 |
| `accelerator` | "gpu"  | 使用GPU训练 |
| `devices` | "auto" | 自动检测可用GPU |

### 数据增强 (data_module.py)
- RandomCrop(32, padding=4)
- RandomHorizontalFlip(p=0.5)
- RandomRotation(15)
- ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
- RandomGrayscale(p=0.2)
- RandomAffine(translate=(0.1, 0.1))
- RandomErasing(p=0.5, scale=(0.02, 0.2))
- Mixup (50%概率, α=0.2)

### 优化器设置 (model.py)
- **优化器**: AdamW with AMSGrad
- **调度器**: CosineAnnealingWarmRestarts (T_0=40, T_mult=2)
- **标签平滑**: 0.1
- **Dropout**: 0.5/0.4
- **权重衰减**: 1e-4
- **梯度裁剪**: 1.0

## 预期训练结果
### 检查点
最佳模型自动保存在 `./checkpoints/`:
- 格式: `cifar10-cnn-epoch={epoch:02d}-val_accuracy={accuracy:.4f}.ckpt`
- 保存前3个模型
- 始终保存最后一个检查点
- 最终模型: `final_model.pth`

## 技术细节
### GPU优化
1. **更大的批次大小**: 64 → 128
   - 利用GPU并行计算
   - 提升训练稳定性
2. **更多工作线程**: 4 → 8
   - 减少数据加载瓶颈
   - 保持GPU数据供应
3. **持久化工作线程**: 启用
   - 避免工作线程初始化开销
   - 加快epoch切换
4. **数据预取**: prefetch_factor=2
   - GPU计算时CPU准备数据
   - 减少GPU空闲时间
5. **学习率缩放**: 线性缩放（随batch size）
   - 公式: lr_new = lr_old × (batch_size_new / batch_size_old)
   - 0.001 × (128/64) = 0.002
6. **梯度裁剪**: 1.0
   - 防止梯度爆炸
   - 提升训练稳定性
### 模型改进
1. **增加容量**: 96→192→384→512 → 128→256→512→768
   - 更多通道用于复杂特征
   - 更好的表示学习
2. **残差连接**: 在深层添加
   - 解决梯度消失
   - 允许更深的网络
3. **Mixup增强**: 50%概率
   - 减少过拟合
   - 提升泛化能力
4. **Warm Restart**: T_0=40, T_mult=2
   - 周期性学习率重启
   - 跳出局部最优

## 项目信息
- **项目**: CIFAR10 CNN分类
- **目标**: 93%+ 测试准确率
- **框架**: PyTorch Lightning
- **技术**: CBAM + ResNet + Mixup + Warm Restart

## 许可证
本项目遵循 [LICENSE](LICENSE) 文件中指定的许可条款。

## 实验记录
| 实验 | Epochs | Batch Size | 验证集准确率 | 备注        |
|----|--------|------------|--------|-----------|
| 1  | 2      | 32         | 64.72% | 快速验证      |
| 2  | 30     | 64         | 86.62% | 完整训练      |
| 3  | 80     | 64         | 89.83% | 进一步完整训练   |
| 4  | 190    | 64         | 90.94% | 微调后完整训练   |
| 5  | 100    | 64         | 91.20% | 优化后完整训练   |
| 6  | 120    | 64         | 90.40% | 再次调整后完整训练 |
| 7  | 150    | 128 (GPU)  | 91.48% | GPU优化版本   |
| 8  | 200    | 128 (GPU)  | 92.00% | GPU优化第二版  |
| 9  | 200    | 128 (GPU)  | 93.60% | GPU优化最终版  |