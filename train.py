"""
主训练脚本
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from model import CIFAR10CNN
from data_module import CIFAR10DataModule
from config import config


def train():
    """训练主函数"""
    print("=" * 60)
    print("CIFAR10 CNN Training - Target: 93%+ Accuracy")
    print("=" * 60)

    # 检查GPU
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU Not Available, using CPU")

    # 创建必要的目录
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 初始化数据模块
    print("\nLoading CIFAR10 dataset...")
    data_module = CIFAR10DataModule()
    data_module.prepare_data()
    data_module.setup()

    # 初始化模型
    print("Initializing CNN model...")
    model = CIFAR10CNN()

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")

    # 定义回调函数
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename='cifar10-cnn-{epoch:02d}-{val_accuracy:.4f}',
        monitor=config.checkpoint_monitor,
        mode=config.checkpoint_mode,
        save_top_k=3,
        save_last=True,
        verbose=True
    )

    # ============ 修改：调整早停参数，给模型更多改进机会 ============
    # 技术原理：增加patience和降低min_delta，允许模型在更长训练周期内缓慢改进
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=40,  # 修改：从30增加到40，给warm restart策略更多时间
        mode='max',
        verbose=True,
        min_delta=0.0003  # 修改：从0.0005降低到0.0003，检测更小的改进
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 初始化TensorBoard日志
    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name='cifar10_cnn'
    )

    # 初始化训练器
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=50,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        precision= 32
    )

    # 开始训练
    print("\nStarting training...")
    print(f"Max Epochs: {config.max_epochs}")
    print(f"Target Accuracy: 93%+")
    print("=" * 60)

    trainer.fit(model, data_module)

    # 测试最佳模型
    print("\nTesting best model...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model: {best_model_path}")
        model = CIFAR10CNN.load_from_checkpoint(best_model_path)

        # 在测试集上评估
        test_results = trainer.test(model, data_module)
        test_accuracy = test_results[0]['test_accuracy']

        print("=" * 60)
        print(f"Final Test Accuracy: {test_accuracy:.4f}")
        if test_accuracy >= 0.93:
            print(f"SUCCESS: Achieved target accuracy (≥93%)!")
        else:
            print(f"Target not reached, but close to {test_accuracy * 100:.2f}%")
        print("=" * 60)

        # 保存最终模型
        torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "final_model.pth"))
        print(f"Model saved to: {os.path.join(config.checkpoint_dir, 'final_model.pth')}")

    print("\nTraining completed!")


def quick_test():
    """快速测试函数，验证代码是否可运行"""
    print("\nRunning quick test...")

    # 测试数据加载
    data_module = CIFAR10DataModule()
    data_module.prepare_data()
    data_module.setup()

    # 测试单个batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    images, labels = batch

    print(f"   Data loaded successfully!")
    print(f"   Batch shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Number of classes: {len(set(labels.numpy()))}")

    # 测试模型
    model = CIFAR10CNN()
    output = model(images)

    print(f"   Model forward pass successful!")
    print(f"   Output shape: {output.shape}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CIFAR10 CNN Training')
    parser.add_argument('--test', action='store_true', help='Run quick test only')

    args = parser.parse_args()

    if args.test:
        quick_test()
    else:
        train()