# 第七次实验
# GPU优化修改总结

## 修改概览

本项目已针对GPU训练进行全面优化，目标是在CIFAR10测试集上达到93%以上的准确率。

## 文件修改清单

### 1. config.py - GPU训练配置

| 参数 | 原值 | 新值 | 修改原因 |
|------|------|------|----------|
| `batch_size` | 64 | 128 | GPU可以处理更大的batch，提升训练效率和稳定性 |
| `num_workers` | 4 | 8 | 充分利用GPU的多线程能力，加速数据加载 |
| `learning_rate` | 0.01 | 0.001 | 配合更大的batch size使用更小的学习率 |
| `max_epochs` | 120 | 150 | 给warm restart策略更多收敛时间 |
| `accelerator` | "cpu" | "gpu" | 启用GPU训练 |
| `devices` | 1 | "auto" | 自动检测可用的GPU数量 |

**技术原理**：
- 学习率线性缩放原则：batch size加倍，学习率相应调整
- 更大的workers减少数据加载瓶颈，充分利用GPU性能

### 2. train.py - 训练脚本优化

| 修改项 | 原值 | 新值 | 修改原因 |
|--------|------|------|----------|
| GPU检测 | 简单检查 | 详细信息输出 | 显示GPU数量、型号和显存 |
| `patience` | 40 | 50 | GPU训练更快，给warm restart更多时间 |
| `min_delta` | 0.0003 | 0.0002 | 更敏感地检测小的改进 |
| `deterministic` | True | False | 允许GPU使用不确定性优化算法 |
| `gradient_clip_val` | 无 | 1.0 | 防止梯度爆炸，提升训练稳定性 |
| `accumulate_grad_batches` | 无 | 1 | 可配置的梯度累积 |

**技术原理**：
- 梯度裁剪防止训练不稳定，特别是在使用更大的学习率时
- 降低deterministic允许使用cuDNN的benchmark模式加速训练

### 3. data_module.py - 数据加载优化

| 修改项 | 原值 | 新值 | 修改原因 |
|--------|------|------|----------|
| `persistent_workers` | False | True | 保持worker进程存活，减少初始化开销 |
| `prefetch_factor` | 默认 | 2 | 每个worker预取2个batch，加速数据加载 |
| `pin_memory` | True | True（保持） | 加速CPU到GPU的数据传输 |

**技术原理**：
- persistent_workers避免每个epoch重新创建worker，显著减少训练开销
- prefetch_factor使GPU在计算时，CPU提前准备下一批数据，减少等待时间

### 4. model.py - 学习率优化

| 修改项 | 原值 | 新值 | 修改原因 |
|--------|------|------|----------|
| 初始学习率 | 0.001 | 0.002 | 配合batch size增加（线性缩放：0.001 * 128/64） |
| `T_0` | 30 | 40 | 适应150 epochs的训练 |
| `amsgrad` | False | True | 使用AMSGrad变体，提升GPU训练稳定性 |

**技术原理**：
- 学习率线性缩放：batch size从64增加到128（2倍），学习率从0.001增加到0.002
- AMSGrad是Adam的改进变体，在某些情况下提供更好的收敛性
- 更长的restart周期（T_0=40）给模型更多时间探索解空间

## 核心技术原理

### 1. 学习率线性缩放
- **原理**：当batch size线性增加时，学习率也应相应增加
- **公式**：lr_new = lr_old * (batch_size_new / batch_size_old)
- **应用**：0.001 * (128/64) = 0.002

### 2. Mixup数据增强
- **原理**：通过线性组合两个样本及其标签创建新的训练样本
- **效果**：减少过拟合，提升泛化能力
- **实现**：50%概率使用Mixup，混合系数α=0.2

### 3. 残差连接
- **原理**：添加跳跃连接解决深层网络的梯度消失问题
- **效果**：允许训练更深的网络而不损失性能
- **实现**：在第三和第四卷积阶段添加ResidualBlock

### 4. Warm Restart学习率调度
- **原理**：周期性重启学习率，帮助模型跳出局部最优
- **参数**：T_0=40（第一个周期40个epoch），T_mult=2（周期倍增）
- **效果**：在训练过程中多次探索不同的学习率范围

### 5. GPU优化技术
- **大batch训练**：利用GPU并行计算能力
- **多worker数据加载**：减少GPU等待数据的时间
- **梯度裁剪**：防止梯度爆炸
- **数据预取**：GPU计算时CPU提前准备数据

## 预期效果

### 训练时间
- **CPU训练**：约10-12小时（100 epochs, batch_size=64）
- **GPU训练**：约1-2小时（150 epochs, batch_size=128）

### 准确率
- **CPU训练**：91.20%（实际达到）
- **GPU训练**：预期93%以上（目标）

### 关键改进
1. 更大的模型容量（通道数增加）
2. 残差连接（解决深度网络问题）
3. Mixup增强（提升泛化能力）
4. Warm Restart（跳出局部最优）
5. GPU加速（更快更稳定）

## 使用步骤

### 1. 上传代码到服务器
```bash
scp -P <端口> -r E:\python_exercises\zms_cifar10_cnn <用户名>@<主机名>:~/projects/
```

### 2. SSH登录服务器
```bash
ssh -p <端口> <用户名>@<主机名>
```

### 3. 安装依赖
```bash
cd ~/projects/zms_cifar10_cnn
pip install -r requirements.txt
```

### 4. 检查GPU
```bash
nvidia-smi
```

### 5. 运行训练
```bash
python train.py
```

### 6. 监控训练
```bash
tensorboard --logdir=./logs
```

## 故障排除

### GPU内存不足
- 减小batch_size（config.py）
- 减少模型通道数（model.py）
- 使用梯度累积（train.py）

### 训练不稳定
- 降低学习率（model.py）
- 增加梯度裁剪强度（train.py）
- 减少Mixup概率（model.py）

### 准确率不达标
- 增加训练轮数（config.py）
- 调整warm restart参数（model.py）
- 尝试不同的数据增强组合（data_module.py）

## 提交清单

提交到GitHub时应包含：
1. ✅ 所有源代码文件（model.py, data_module.py, config.py, train.py等）
2. ✅ GPU_TRAINING_GUIDE.md（GPU训练说明）
3. ✅ SSH_GUIDE.md（SSH操作指南）
4. ✅ 本修改总结文件
5. ⏳ 训练日志和checkpoint文件（训练完成后）
6. ⏳ README文件说明最终结果和准确率（训练完成后）

## 成功标准

✅ 在CIFAR10测试集上达到93%以上的准确率
✅ 代码在GPU服务器上正常运行
✅ 提交到GitHub的public repository
✅ 发送repository链接给修博士
