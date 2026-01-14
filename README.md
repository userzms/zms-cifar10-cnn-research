# CIFAR10 CNN Classification Project

## 项目目标
使用PyTorch Lightning构建CNN模型，在CIFAR10数据集上达到93%以上的测试准确率。

## 项目结构
```
cifar10_cnn/
├── train.py          # 主训练脚本
├── model.py          # CNN模型定义
├── data_module.py    # 数据加载模块
├── config.py         # 配置参数
├── requirements.txt  # 依赖包
└── README.md         # 项目说明
```


## 环境要求
- Python 3.13.11
- PyTorch 2.9.1+cpu
- PyTorch Lightning 2.6.0
- Torchvision 0.24.1+cpu
- TorchMetrics 1.8.2
- TensorBoard 2.20.0
- Matplotlib 3.10.8
- NumPy 2.4.1

## 安装依赖
```bash
pip install -r requirements.txt