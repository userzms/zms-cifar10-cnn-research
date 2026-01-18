# AGENTS.md

## Project Overview
PyTorch Lightning CNN for CIFAR10 classification targeting 93%+ test accuracy.
Tech stack: PyTorch 2.0+, Lightning 2.0+, TorchMetrics 1.4+, TensorBoard.

## Essential Commands

```bash
# Run full training
python train.py

# Quick test (validates data loading & model forward pass)
python train.py --test

# Verify dataset integrity
python detailed_verify.py

# Install dependencies
pip install -r requirements.txt

# View training logs in TensorBoard
tensorboard --logdir=./logs
```

**Note**: No testing framework (pytest, unittest) or linters (ruff, black, pylint, mypy) are configured in this project.

## Code Style Guidelines

### Imports
Order: Standard library → third-party → local imports. Use blank lines between groups.

```python
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy

from model import CIFAR10CNN
from data_module import CIFAR10DataModule
from config import config
```

### Indentation
- 4 spaces (no tabs)

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `CIFAR10CNN`, `CBAM`, `CIFAR10DataModule`)
- **Functions/Methods**: `snake_case` (e.g., `train()`, `forward()`, `setup()`)
- **Config Attributes**: `lowercase_with_underscores` (e.g., `batch_size`, `learning_rate`)
- **Constants**: `UPPERCASE` (not commonly used in this codebase)

### Type Hints
Not used - dynamic typing throughout the codebase.

### Comments/Docstrings
- Chinese language for all comments and docstrings
- Triple-quoted strings for module-level docstrings

```python
"""
CNN模型定义 - 专门为CIFAR10设计
"""
```

### Error Handling
Minimal error handling - relies on PyTorch Lightning's built-in error handling and validation.

### File Organization
Single responsibility principle per file:
- `model.py`: CNN model architecture with CBAM attention
- `data_module.py`: Data loading and augmentation
- `config.py`: Centralized configuration
- `train.py`: Training loop and callbacks

## Project-Specific Patterns

### Config Class
Centralized `Config` class in `config.py` with singleton `config` instance:
```python
class Config:
    data_dir = "./data"
    batch_size = 64
    learning_rate = 0.01
    num_classes = 10

config = Config()
```

### LightningModule
Extend `pl.LightningModule` and implement:
- `forward(self, x)`: Forward pass
- `training_step(self, batch, batch_idx)`: Training logic
- `validation_step(self, batch, batch_idx)`: Validation logic
- `test_step(self, batch, batch_idx)`: Test logic
- `configure_optimizers(self)`: Optimizer and scheduler configuration
- `on_train_epoch_end(self)`: Epoch-end hooks

### LightningDataModule
Extend `pl.LightningDataModule` and implement:
- `prepare_data()`: Download data
- `setup(self, stage)`: Split datasets
- `train_dataloader()`: Training data loader
- `val_dataloader()`: Validation data loader
- `test_dataloader()`: Test data loader

### Metrics
Use `torchmetrics.Accuracy`:
```python
from torchmetrics import Accuracy

self.train_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
```

### Callbacks
Standard PyTorch Lightning callbacks:
- `ModelCheckpoint`: Save best models
- `EarlyStopping`: Stop training if no improvement
- `LearningRateMonitor`: Track learning rate

### Data Augmentation
Comprehensive transforms in `data_module.py`:
- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(15)`
- `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)`
- `RandomGrayscale(p=0.2)`
- `RandomErasing(p=0.5, scale=(0.02, 0.2))`
- Normalization: Mean=(0.4914, 0.4822, 0.4465), Std=(0.2470, 0.2435, 0.2616)

### Model Architecture
Current architecture in `model.py`:
- 4 convolutional blocks (96→192→384→512 channels)
- Each block: Conv2d → BatchNorm2d → ReLU → CBAM attention → MaxPool2d
- Global Average Pooling
- 3 fully connected layers (384→192→10) with dropout (0.4, 0.3)
- Label smoothing: 0.1

### Optimizer & Scheduler
```python
optimizer = AdamW(lr=0.01, weight_decay=1e-4, betas=(0.9, 0.999))
scheduler = OneCycleLR(max_lr=0.01, epochs=max_epochs, pct_start=0.3, anneal_strategy='cos')
```

## Development Notes

- No existing tests or linting configured
- Chinese comments/docstrings throughout codebase
- **Logs**: `./logs/cifar10_cnn/` (TensorBoard format)
- **Checkpoints**: `./checkpoints/` (format: `cifar10-cnn-{epoch:02d}-{val_accuracy:.4f}`)
- **Data**: `./data/` (auto-downloaded CIFAR10)
- **Training parameters**: batch_size=64, max_epochs=100, learning_rate=0.01
- **Target accuracy**: 93%+ test accuracy
- **Precision**: 32-bit floating point
- **Early stopping**: patience=30, min_delta=0.0005
- **Checkpoint monitoring**: `val_accuracy` (maximize)

## Quick Reference

| File | Purpose |
|------|---------|
| `train.py` | Main training script with `train()` and `quick_test()` functions |
| `model.py` | CNN model with CBAM attention (CIFAR10CNN class) |
| `data_module.py` | CIFAR10 data loading and augmentation (CIFAR10DataModule class) |
| `config.py` | Centralized configuration (Config class) |
| `detailed_verify.py` | Dataset verification script |
