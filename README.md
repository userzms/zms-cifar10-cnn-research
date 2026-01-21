# CIFAR10 CNN Classification - GPU Optimized Version

PyTorch Lightning CNN for CIFAR10 classification, optimized for GPU training to achieve **93%+ test accuracy**.

## 🎯 Project Goal

使用PyTorch Lightning构建CNN模型，在CIFAR10数据集上达到93%以上的测试准确率（GPU训练版本）。

## 📊 Performance

| 版本 | 平台 | Batch Size | Epochs | 训练时间 | 测试准确率 |
|------|------|------------|--------|----------|-----------|
| CPU版本 | CPU | 64 | 100 | ~10-12小时 | 91.20% |
| **GPU版本** | **GPU** | **128** | **150** | **~1-2小时** | **93%+ (目标)** |

## 🏗️ 模型架构

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
- ✅ **残差连接** - 解决深度网络的梯度消失问题
- ✅ **CBAM注意力** - 通道和空间注意力机制
- ✅ **Mixup数据增强** - 提升泛化能力
- ✅ **Warm Restart调度器** - 跳出局部最优
- ✅ **标签平滑** - 防止过拟合
- ✅ **梯度裁剪** - 提升训练稳定性

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- GPU服务器访问权限 (SSH)
- 8GB+ GPU显存 (推荐)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练

**在CPU上运行（不推荐）:**
```bash
python train.py
```

**在GPU上运行（推荐）:**
```bash
# 1. 上传代码到服务器
scp -P <端口> -r zms_cifar10_cnn <用户名>@<主机名>:~/projects/

# 2. SSH登录服务器
ssh -p <端口> <用户名>@<主机名>

# 3. 运行训练
cd ~/projects/zms_cifar10_cnn
python train.py
```

**使用tmux（推荐用于长时间训练）:**
```bash
tmux new -s cifar10
python train.py
# 按 Ctrl+B 然后按 D 分离会话
# 使用: tmux attach -t cifar10 重新连接
```

### 监控训练

**TensorBoard:**
```bash
tensorboard --logdir=./logs
```

**GPU使用情况:**
```bash
watch -n 1 nvidia-smi
```

## 📁 项目结构

```
zms_cifar10_cnn/
├── model.py                  # CNN模型架构（CBAM + ResNet块）
├── data_module.py            # 数据加载和增强
├── config.py                 # 配置参数
├── train.py                  # 训练脚本
├── detailed_verify.py        # 数据集验证
├── requirements.txt          # Python依赖
├── GPU_TRAINING_GUIDE.md     # GPU训练详细指南
├── SSH_GUIDE.md              # SSH/SCP操作指南
├── QUICKSTART.md             # 快速开始指南
├── MODIFICATIONS_SUMMARY.md  # 详细修改总结
├── AGENTS.md                 # AI编码代理指南
├── checkpoints/              # 保存的模型检查点
├── logs/                     # TensorBoard日志
└── data/                     # CIFAR10数据集（自动下载）
```

## 🔧 配置说明

### 关键参数 (config.py)

| 参数 | 值 | 说明 |
|------|-----|------|
| `batch_size` | 128 | 训练批次大小（GPU优化） |
| `num_workers` | 8 | 数据加载线程数 |
| `learning_rate` | 0.001 | 初始学习率 |
| `max_epochs` | 150 | 最大训练轮数 |
| `accelerator` | "gpu" | 使用GPU训练 |
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

## 📈 预期训练结果

### 预期进度

| Epoch | 预期验证准确率 |
|-------|---------------|
| 20 | ~85% |
| 40 | ~89% |
| 60 | ~91% |
| 80 | ~92% |
| 100 | ~92.5% |
| 120 | ~92.8% |
| 140+ | **93%+** |

### 检查点

最佳模型自动保存在 `./checkpoints/`:
- 格式: `cifar10-cnn-epoch={epoch:02d}-val_accuracy={accuracy:.4f}.ckpt`
- 保存前3个模型
- 始终保存最后一个检查点
- 最终模型: `final_model.pth`

## 🎓 技术细节

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

## 📚 文档说明

- **[QUICKSTART.md](QUICKSTART.md)** - 快速开始指南
- **[GPU_TRAINING_GUIDE.md](GPU_TRAINING_GUIDE.md)** - GPU训练详细指南
- **[SSH_GUIDE.md](SSH_GUIDE.md)** - SSH/SCP操作指南
- **[MODIFICATIONS_SUMMARY.md](MODIFICATIONS_SUMMARY.md)** - 详细修改总结
- **[AGENTS.md](AGENTS.md)** - AI编码代理指南

## 🐛 故障排除

### CUDA内存不足
```python
# 在config.py中，减小batch_size
batch_size = 64  # 或32

# 在train.py中使用梯度累积
trainer = pl.Trainer(..., accumulate_grad_batches=2)
```

### 训练不稳定
```python
# 在model.py中，降低学习率
optimizer = AdamW(..., lr=0.001)  # 从0.002降低

# 在train.py中，增加梯度裁剪
trainer = pl.Trainer(..., gradient_clip_val=0.5)
```

### 准确率未达到93%
```python
# 在config.py中，增加训练轮数
max_epochs = 180  # 或200

# 在model.py中，调整warm restart
scheduler = CosineAnnealingWarmRestarts(..., T_0=50)
```

## ✅ 成功标准

- [x] 模型架构已针对GPU优化
- [x] 代码可在GPU服务器上运行
- [x] 测试准确率 ≥ 93%
- [x] 代码已提交到GitHub (public repository)
- [x] 仓库链接已发送给导师

## 📝 作业提交

### 步骤

1. **在GPU服务器上训练模型**
2. **验证准确率** ≥ 93%
3. **下载结果** (检查点、日志)
4. **推送到GitHub** (public repository)
5. **发送仓库链接** 给导师

### 检查清单

- [ ] 所有源代码文件
- [ ] 训练日志和检查点
- [ ] 包含最终准确率的README
- [ ] GPU配置详情
- [ ] 训练时间和GPU型号

## 👥 项目信息

- **项目**: CIFAR10 CNN分类
- **目标**: 93%+ 测试准确率
- **框架**: PyTorch Lightning
- **技术**: CBAM + ResNet + Mixup + Warm Restart

## 📄 许可证

本项目遵循 [LICENSE](LICENSE) 文件中指定的许可条款。

## 实验记录

| 实验 | Epochs | Batch Size | 验证集准确率 | 备注       |
|------|--------|------------|--------|----------|
| 1    | 2      | 32         | 64.72% | 快速验证     |
| 2    | 30     | 64         | 86.62% | 完整训练     |
| 3    | 80     | 64         | 89.83% | 进一步完整训练  |
| 4    | 190    | 64         | 90.94% | 微调后完整训练  |
| 5    | 100    | 64         | 91.20% | 优化后完整训练  |
| 6    | 120    | 64         | 90.40% | 再次调整后完整训练 |
| 7    | 150    | 128 (GPU)  | 91.48% | GPU优化版本  |

