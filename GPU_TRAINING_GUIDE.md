# GPU训练说明

## 项目优化概述

本项目已针对GPU训练进行优化，目标是在CIFAR10测试集上达到93%以上的准确率。

## 主要优化内容

### 1. GPU训练配置 (config.py)
- **accelerator**: 从"cpu"改为"gpu"，启用GPU训练
- **devices**: 从1改为"auto"，自动检测可用的GPU数量
- **batch_size**: 从64增加到128，GPU可以处理更大的batch，提升训练效率和稳定性
- **num_workers**: 从4增加到8，充分利用GPU的多线程能力
- **learning_rate**: 从0.01降低到0.001，配合更大的batch size使用更小的学习率
- **max_epochs**: 从120增加到150，给warm restart策略更多收敛时间

**技术原理**：
- 更大的batch size提供更稳定的梯度估计，但需要相应调整学习率（线性缩放原则）
- GPU的并行计算能力允许更大的batch，显著加快训练速度
- 更多的workers减少数据加载瓶颈，充分利用GPU性能

### 2. 训练脚本优化 (train.py)
- 增强GPU检测和信息输出，显示GPU数量、型号和显存
- 优化早停参数：patience从40增加到50，min_delta从0.0003降低到0.0002
- 设置`deterministic=False`以允许GPU使用不确定性优化算法提升性能
- 添加梯度裁剪（gradient_clip_val=1.0）防止梯度爆炸
- 添加梯度累积选项（accumulate_grad_batches=1）

**技术原理**：
- GPU训练更快，需要增加patience给warm restart策略充分时间
- 降低min_delta更敏感地检测小的改进
- 梯度裁剪提升训练稳定性，特别是在使用更大的学习率时

### 3. 数据加载优化 (data_module.py)
- 添加`persistent_workers=True`保持worker进程存活，减少epoch间worker初始化开销
- 添加`prefetch_factor=2`每个worker预取2个batch，加速数据加载
- 保持`pin_memory=True`加速CPU到GPU的数据传输

**技术原理**：
- persistent_workers避免每个epoch重新创建worker，显著减少训练开销
- prefetch_factor使GPU在计算时，CPU提前准备下一批数据，减少等待时间

### 4. 学习率优化 (model.py)
- 调整初始学习率从0.001到0.002，配合batch size增加（线性缩放：0.001 * 128/64）
- 调整warm restart参数：T_0从30增加到40，适应150 epochs的训练
- 添加`amsgrad=True`使用AMSGrad变体，提升GPU训练稳定性

**技术原理**：
- 学习率线性缩放：batch size加倍，学习率也相应加倍
- AMSGrad是Adam的改进变体，在某些情况下提供更好的收敛性
- 更长的restart周期给模型更多时间探索解空间

## 在GPU服务器上运行

### 1. 上传代码到GPU服务器

使用scp命令上传代码到服务器：
```bash
scp -P <端口> -r <本地项目目录> <用户名>@<主机名>:<目标目录>
```

示例：
```bash
scp -P 22 -r zms_cifar10_cnn user@server.example.com:~/projects/
```

### 2. SSH登录GPU服务器

```bash
ssh -p <端口> <用户名>@<主机名>
```

示例：
```bash
ssh -p 22 user@server.example.com
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

或者运行Python检查：
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 5. 运行训练

```bash
python train.py
```

或使用多GPU（如果可用）：
```bash
python train.py --devices "0,1"  # 使用GPU 0和1
```

### 6. 监控训练

使用TensorBoard查看训练进度：
```bash
tensorboard --logdir=./logs
```

## 训练时间估计

- **CPU训练**：约10-12小时（100 epochs, batch_size=64）
- **GPU训练**：约1-2小时（150 epochs, batch_size=128）

**注意**：实际时间取决于GPU型号和数量。

## 预期结果

基于优化后的配置，预期在CIFAR10测试集上达到93%以上的准确率。

### 关键优化指标

1. **模型容量增加**：通道数从96→192→384→512增加到128→256→512→768
2. **残差连接**：添加ResNet风格的跳跃连接，解决深度网络梯度消失问题
3. **Mixup数据增强**：50%概率使用Mixup，显著提升泛化能力
4. **Warm Restart**：CosineAnnealingWarmRestarts帮助模型跳出局部最优
5. **GPU加速**：更大的batch size和更多的workers充分利用GPU并行能力

## 故障排除

### GPU不可用

如果遇到"CUDA out of memory"错误：
1. 减小batch_size（config.py中的batch_size）
2. 减少模型通道数（model.py中的通道配置）
3. 使用梯度累积（train.py中的accumulate_grad_batches）

### 训练不稳定

如果损失爆炸或不收敛：
1. 降低学习率（model.py中的lr）
2. 增加梯度裁剪强度（train.py中的gradient_clip_val）
3. 减少Mixup概率（model.py中的mixup_alpha）

### 准确率不达标

如果准确率未达到93%：
1. 增加训练轮数（config.py中的max_epochs）
2. 调整warm restart参数（model.py中的T_0）
3. 尝试不同的数据增强组合（data_module.py中的transforms）

## 提交作业

训练完成后，将以下文件提交到GitHub：
1. 所有源代码文件（model.py, data_module.py, config.py, train.py等）
2. 训练日志和checkpoint文件
3. README文件说明训练结果和准确率

确保创建public repository并发送链接给修博士。
