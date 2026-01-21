# GPU训练快速开始指南

## 🎯 目标

在CIFAR10测试集上达到93%以上的准确率，使用GPU加速训练。

## 📋 准备工作

### 服务器信息
确保你已经从修博士那里获得了以下信息：
- ✅ Host (主机名/IP地址)
- ✅ Port (SSH端口)
- ✅ User (用户名)
- ✅ Password (密码) 或 SSH密钥

### 本地环境
- ✅ 已安装Git
- ✅ 已安装Python (3.8+)
- ✅ 项目代码已优化为GPU版本

## 🚀 快速开始

### 步骤1: 上传代码到服务器

```bash
scp -P <端口> -r E:\python_exercises\zms_cifar10_cnn <用户名>@<主机名>:~/projects/
```

**示例：**
```bash
scp -P 22 -r E:\python_exercises\zms_cifar10_cnn username@server.example.com:~/projects/
```

### 步骤2: SSH登录服务器

```bash
ssh -p <端口> <用户名>@<主机名>
```

**示例：**
```bash
ssh -p 22 username@server.example.com
```

### 步骤3: 检查GPU

```bash
nvidia-smi
```

预期输出示例：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.73.01    Driver Version: 460.73.01    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 3090    Off  | 00000000:01:00.0 Off |                  N/A |
| 34%   42C    P2    62W / 350W |      4MiB / 24264MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### 步骤4: 安装依赖

```bash
cd ~/projects/zms_cifar10_cnn
pip install -r requirements.txt
```

### 步骤5: 运行训练

**方法1: 前台运行（直接输出到终端）**
```bash
python train.py
```

**方法2: 后台运行（推荐，使用tmux）**
```bash
# 创建tmux会话
tmux new -s cifar10

# 运行训练
cd ~/projects/zms_cifar10_cnn
python train.py

# 分离会话：按 Ctrl+B 然后按 D

# 重新连接会话
tmux attach -t cifar10
```

**方法3: 后台运行（输出到日志文件）**
```bash
nohup python train.py > train.log 2>&1 &

# 查看日志
tail -f train.log
```

### 步骤6: 监控训练

**查看训练进度（使用TensorBoard）**
```bash
# 在另一个终端登录服务器
ssh -p <端口> <用户名>@<主机名>

# 启动TensorBoard
cd ~/projects/zms_cifar10_cnn
tensorboard --logdir=./logs --port=6006

# 在本地浏览器访问：http://<服务器IP>:6006
```

**查看GPU使用情况**
```bash
watch -n 1 nvidia-smi
```

**查看训练日志**
```bash
tail -f train.log
```

### 步骤7: 下载训练结果

训练完成后，下载checkpoint和日志文件：

```bash
scp -P <端口> -r <用户名>@<主机名>:~/projects/zms_cifar10_cnn/checkpoints E:\python_exercises\zms_cifar10_cnn\
```

**示例：**
```bash
scp -P 22 -r username@server.example.com:~/projects/zms_cifar10_cnn/checkpoints E:\python_exercises\zms_cifar10_cnn\
```

## 📊 预期结果

### 训练时间
- **GPU训练**：约1-2小时（150 epochs, batch_size=128）

### 准确率目标
- **目标**：93%以上
- **当前CPU版本**：91.20%

### 关键优化
1. ✅ GPU加速（batch_size=128）
2. ✅ 更大模型容量
3. ✅ 残差连接
4. ✅ Mixup数据增强
5. ✅ Warm Restart学习率调度

## ⚙️ 配置说明

### config.py - 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 128 | GPU训练的批次大小 |
| num_workers | 8 | 数据加载的线程数 |
| learning_rate | 0.001 | 初始学习率 |
| max_epochs | 150 | 最大训练轮数 |
| accelerator | "gpu" | 使用GPU训练 |
| devices | "auto" | 自动检测GPU数量 |

### 遇到GPU内存不足？

如果遇到"CUDA out of memory"错误，修改config.py：

```python
# 方法1: 减小batch_size
batch_size = 64  # 从128改为64

# 方法2: 在train.py中使用梯度累积
trainer = pl.Trainer(
    ...
    accumulate_grad_batches=2  # 将有效batch_size增加到256
)
```

### 遇到训练不稳定？

如果损失爆炸或不收敛，修改model.py：

```python
# 方法1: 降低学习率
optimizer = AdamW(..., lr=0.001)  # 从0.002改为0.001

# 方法2: 增加梯度裁剪
trainer = pl.Trainer(..., gradient_clip_val=0.5)  # 从1.0改为0.5

# 方法3: 减少Mixup概率
if torch.rand(1).item() < 0.3:  # 从0.5改为0.3
```

## 🎯 验证成功标准

训练完成后，检查输出中的以下信息：

```bash
============================================================
Final Test Accuracy: 0.93XX (93.XX%)
SUCCESS: Achieved target accuracy (≥93%)!
============================================================
```

如果看到"SUCCESS"消息，恭喜你成功完成了任务！

## 📝 提交作业

### 1. 提交到GitHub

```bash
# 在本地
git add .
git commit -m "GPU训练版本 - 达到93%准确率"
git push origin main
```

### 2. 创建Public Repository

1. 登录GitHub.com
2. 创建新的public repository
3. 推送代码到GitHub
4. 确保所有文件都在仓库中

### 3. 发送链接给修博士

- ✅ Repository URL
- ✅ 测试准确率
- ✅ 训练时间
- ✅ 使用的GPU型号

## 🔧 常用命令速查

| 操作 | 命令 |
|------|------|
| SSH登录 | `ssh -p 22 user@server` |
| 上传代码 | `scp -P 22 -r local_dir user@server:~/dest` |
| 下载文件 | `scp -P 22 user@server:~/remote_file local_dest` |
| 查看GPU | `nvidia-smi` |
| 后台运行 | `nohup python train.py > train.log 2>&1 &` |
| 查看日志 | `tail -f train.log` |
| tmux创建 | `tmux new -s cifar10` |
| tmux分离 | `Ctrl+B 然后按 D` |
| tmux连接 | `tmux attach -t cifar10` |

## 📚 详细文档

- `GPU_TRAINING_GUIDE.md` - GPU训练详细说明
- `SSH_GUIDE.md` - SSH和SCP完整操作指南
- `MODIFICATIONS_SUMMARY.md` - 所有修改的详细总结

## ❓ 遇到问题？

1. **GPU不可用**：检查CUDA驱动和PyTorch版本
2. **SSH连接失败**：确认Host、Port、User、Password正确
3. **上传失败**：检查本地路径和服务器路径
4. **训练崩溃**：查看train.log中的错误信息
5. **准确率不达标**：增加训练轮数或调整学习率

## 🎉 祝你成功！

如果达到93%以上准确率，恭喜你成功完成任务！记得提交到GitHub并发送链接给修博士。

---

**最后更新**: 2026-01-19
**版本**: GPU Optimized v1.0
**目标**: 93%+ Test Accuracy on CIFAR10
