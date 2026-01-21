# GPU服务器操作指南

## 1. SSH登录服务器

### 基本语法
```bash
ssh -p <端口> <用户名>@<主机名>
```

### 示例
```bash
ssh -p 22 username@server.example.com
```

### 第一次登录
第一次登录时会提示：
```
The authenticity of host 'server.example.com' can't be established.
RSA key fingerprint is SHA256:...
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```
输入 `yes` 并回车，然后输入密码。

### 使用SSH密钥（可选，避免每次输入密码）
```bash
# 生成SSH密钥（如果还没有）
ssh-keygen -t rsa -b 4096

# 将公钥复制到服务器
ssh-copy-id -p <端口> <用户名>@<主机名>
```

## 2. 上传代码到服务器

### 使用SCP上传整个项目

```bash
scp -P <端口> -r <本地项目路径> <用户名>@<主机名>:<服务器目标路径>
```

### 示例

**上传到用户主目录：**
```bash
scp -P 22 -r E:\python_exercises\zms_cifar10_cnn username@server.example.com:~/
```

**上传到特定目录：**
```bash
scp -P 22 -r E:\python_exercises\zms_cifar10_cnn username@server.example.com:~/projects/
```

### 使用SCP上传单个文件

```bash
scp -P <端口> <本地文件路径> <用户名>@<主机名>:<服务器目标路径>
```

### 示例
```bash
scp -P 22 E:\python_exercises\zms_cifar10_cnn\train.py username@server.example.com:~/projects/
```

## 3. 从服务器下载文件

### 下载整个目录

```bash
scp -P <端口> -r <用户名>@<主机名>:<服务器路径> <本地目标路径>
```

### 示例
```bash
scp -P 22 -r username@server.example.com:~/projects/zms_cifar10_cnn/checkpoints E:\python_exercises\zms_cifar10_cnn\
```

### 下载单个文件

```bash
scp -P <端口> <用户名>@<主机名>:<服务器文件路径> <本地目标路径>
```

### 示例
```bash
scp -P 22 username@server.example.com:~/projects/zms_cifar10_cnn/checkpoints/final_model.pth E:\python_exercises\zms_cifar10_cnn\
```

## 4. 服务器端操作

### 检查GPU状态

```bash
nvidia-smi
```

输出示例：
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

### 安装Python依赖

```bash
cd ~/projects/zms_cifar10_cnn
pip install -r requirements.txt
```

### 运行训练

**前台运行（直接输出到终端）：**
```bash
python train.py
```

**后台运行（输出到nohup.out文件）：**
```bash
nohup python train.py > train.log 2>&1 &
```

**使用tmux保持会话（推荐）：**
```bash
# 创建新会话
tmux new -s cifar10

# 运行训练
cd ~/projects/zms_cifar10_cnn
python train.py

# 分离会话（Ctrl+B 然后按 D）

# 重新连接会话
tmux attach -t cifar10

# 列出所有会话
tmux ls
```

### 监控训练进度

**查看训练日志：**
```bash
tail -f train.log
```

**查看GPU使用情况：**
```bash
watch -n 1 nvidia-smi
```

**查看进程：**
```bash
ps aux | grep python
```

### 停止训练

**前台运行的程序：**
```bash
Ctrl+C
```

**后台运行的程序：**
```bash
# 找到进程ID
ps aux | grep train.py

# 终止进程
kill <进程ID>
# 或强制终止
kill -9 <进程ID>
```

## 5. 使用rsync同步文件（推荐）

rsync比scp更高效，支持增量传输，可以中断后继续。

### 上传文件到服务器

```bash
rsync -avz -e "ssh -p <端口>" <本地路径> <用户名>@<主机名>:<服务器路径>
```

### 示例
```bash
rsync -avz -e "ssh -p 22" E:\python_exercises\zms_cifar10_cnn username@server.example.com:~/projects/
```

### 从服务器下载文件

```bash
rsync -avz -e "ssh -p <端口>" <用户名>@<主机名>:<服务器路径> <本地路径>
```

### 示例
```bash
rsync -avz -e "ssh -p 22" username@server.example.com:~/projects/zms_cifar10_cnn/checkpoints E:\python_exercises\zms_cifar10_cnn\
```

## 6. 常见问题

### 连接超时
```bash
ssh -p <端口> -o ConnectTimeout=60 <用户名>@<主机名>
```

### 保持SSH连接不断
在`~/.ssh/config`文件中添加：
```
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

### 权限错误
```bash
chmod +x train.py
```

### 端口被占用
如果SSH默认端口（22）不可用，使用其他端口。

## 7. 完整工作流程示例

```bash
# 1. 上传代码到服务器
scp -P 22 -r E:\python_exercises\zms_cifar10_cnn username@server.example.com:~/projects/

# 2. 登录服务器
ssh -p 22 username@server.example.com

# 3. 检查GPU
nvidia-smi

# 4. 安装依赖
cd ~/projects/zms_cifar10_cnn
pip install -r requirements.txt

# 5. 使用tmux运行训练
tmux new -s cifar10
python train.py
# Ctrl+B 然后按 D 分离会话

# 6. 退出服务器
exit

# 7. 后续重新连接查看训练
ssh -p 22 username@server.example.com
tmux attach -t cifar10

# 8. 训练完成后下载结果
scp -P 22 -r username@server.example.com:~/projects/zms_cifar10_cnn/checkpoints E:\python_exercises\zms_cifar10_cnn\
```

## 8. 快速参考

| 操作 | 命令 |
|------|------|
| 登录服务器 | `ssh -p 22 user@server` |
| 上传目录 | `scp -P 22 -r local_dir user@server:~/dest` |
| 下载目录 | `scp -P 22 -r user@server:~/remote_dir local_dest` |
| 上传文件 | `scp -P 22 local_file user@server:~/dest` |
| 下载文件 | `scp -P 22 user@server:~/remote_file local_dest` |
| 查看GPU | `nvidia-smi` |
| 后台运行 | `nohup python train.py > train.log 2>&1 &` |
| 查看日志 | `tail -f train.log` |
| tmux创建会话 | `tmux new -s session_name` |
| tmux分离会话 | `Ctrl+B 然后按 D` |
| tmux连接会话 | `tmux attach -t session_name` |
