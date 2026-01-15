# CIFAR10 CNN 项目训练日志
## 实验1：初步测试
- 时间：2026年1月14日
- 配置：batch_size=32, max_epochs=2
- 结果：测试准确率 64.71%
- 命令行输出
```bash
(.venv) PS E:\python_exercises\zms_cifar10_cnn> python train.py
============================================================
CIFAR10 CNN Training - Target: 93%+ Accuracy
============================================================
GPU Not Available, using CPU

Loading CIFAR10 dataset...
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\torchvision\datasets\cifar.py:83: VisibleDeprecationWarning: dtype(): align should be passed as Python or NumPy boolean but got `align=0`. Did you mean to pass a tuple to create a subarray type? (Deprecated NumPy 2.4)
  entry = pickle.load(f, encoding="latin1")
Initializing CNN model...
Model Parameters: Total=1,717,450, Trainable=1,717,450
GPU available: False, used: False
TPU available: False, using: 0 TPU cores

Starting training...
Max Epochs: 2
Target Accuracy: 93%+
============================================================

   | Name      | Type               | Params | Mode  | FLOPs
------------------------------------------------------------------
0  | conv1     | Conv2d             | 1.7 K  | train | 0    
1  | bn1       | BatchNorm2d        | 128    | train | 0    
2  | conv2     | Conv2d             | 73.7 K | train | 0    
3  | bn2       | BatchNorm2d        | 256    | train | 0    
4  | conv3     | Conv2d             | 294 K  | train | 0    
5  | bn3       | BatchNorm2d        | 512    | train | 0    
6  | conv4     | Conv2d             | 1.2 M  | train | 0    
7  | bn4       | BatchNorm2d        | 1.0 K  | train | 0    
8  | gap       | AdaptiveAvgPool2d  | 0      | train | 0    
9  | fc1       | Linear             | 131 K  | train | 0    
10 | dropout1  | Dropout            | 0      | train | 0    
11 | fc2       | Linear             | 32.9 K | train | 0    
12 | dropout2  | Dropout            | 0      | train | 0    
13 | fc3       | Linear             | 1.3 K  | train | 0    
14 | criterion | CrossEntropyLoss   | 0      | train | 0    
15 | train_acc | MulticlassAccuracy | 0      | train | 0    
16 | val_acc   | MulticlassAccuracy | 0      | train | 0    
17 | test_acc  | MulticlassAccuracy | 0      | train | 0    
------------------------------------------------------------------
1.7 M     Trainable params
0         Non-trainable params
1.7 M     Total params
6.870     Total estimated model params size (MB)
18        Modules in train mode
0         Modules in eval mode
0         Total Flops
Sanity Checking: |                                                                               | 0/? [00:00<?, ?it/s]E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=17` in the `DataLoader` to improve performance.
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\torch\utils\data\dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=17` in the `DataLoader` to improve performance.
Epoch 0:   6%|█▍                         | 86/1563 [00:06<01:44, 14.19it/s, v_num=0, train_loss=1.800, train_acc=0.281]E
Epoch 0: 100%|█| 1563/1563 [01:57<00:00, 13.25it/s, v_num=0, train_loss=1.590, train_acc=0.688, val_loss=1.420, val_accMetric val_accuracy improved. New best score: 0.494                                                                     
Epoch 0, global step 1563: 'val_accuracy' reached 0.49430 (best 0.49430), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=00-val_accuracy=0.4943.ckpt' as top 3
Epoch 1: 100%|█| 1563/1563 [01:57<00:00, 13.34it/s, v_num=0, train_loss=0.799, train_acc=0.750, val_loss=0.985, val_accMetric val_accuracy improved by 0.153 >= min_delta = 0.001. New best score: 0.647                                       
Epoch 1, global step 3126: 'val_accuracy' reached 0.64710 (best 0.64710), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=01-val_accuracy=0.6471.ckpt' as top 3
`Trainer.fit` stopped: `max_epochs=2` reached.
Epoch 1: 100%|█| 1563/1563 [01:57<00:00, 13.32it/s, v_num=0, train_loss=0.799, train_acc=0.750, val_loss=0.985, val_acc

Testing best model...
Loading best model: E:\python_exercises\zms_cifar10_cnn\checkpoints\cifar10-cnn-epoch=01-val_accuracy=0.6471.ckpt
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=17` in the `DataLoader` to improve performance.
Testing DataLoader 0: 100%|██████████████████████████████████████████████████████████| 313/313 [00:12<00:00, 24.20it/s]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      test_accuracy         0.6470999717712402
        test_loss            0.984864354133606
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
============================================================
Final Test Accuracy: 0.6471
Target not reached, but close to 64.71%
============================================================
Model saved to: ./checkpoints\final_model.pth

Training completed!
```

## 实验2：完整测试
- 时间：2026年1月14日
- 配置：batch_size=64, max_epochs=30
- 结果：测试准确率 86.62%
- 性能分析  
起始准确率：49.56%（Epoch 0）  
中期突破：80.14%（Epoch 10，+30.58%）  
稳定提升：85.53%（Epoch 20，+5.39%）  
最佳性能：86.62%（Epoch 27，+1.09%）  
- 总提升：37.06个百分点
- 后续优化方向   
增加训练epoch至80-100  
轻微增加模型容量  
调整学习率策略  
目标：93%+准确率
- 命令行输出
```bash
(.venv) PS E:\python_exercises\zms_cifar10_cnn> python train.py
============================================================
CIFAR10 CNN Training - Target: 93%+ Accuracy
============================================================
GPU Not Available, using CPU

Loading CIFAR10 dataset...
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\torchvision\datasets\cifar.py:83: VisibleDeprecationWarning: dtype(): align should be passed as Python or NumPy boolean but got `align=0`. Did you mean to pass a tuple to create a subarray type? (Deprecated NumPy 2.4)
  entry = pickle.load(f, encoding="latin1")
Initializing CNN model...
Model Parameters: Total=1,717,450, Trainable=1,717,450
GPU available: False, used: False
TPU available: False, using: 0 TPU cores

Starting training...
Max Epochs: 30
Target Accuracy: 93%+
============================================================
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:881: Checkpoint directory E:\python_exercises\zms_cifar10_cnn\checkpoints exists and is not empty.

   | Name      | Type               | Params | Mode  | FLOPs
------------------------------------------------------------------
0  | conv1     | Conv2d             | 1.7 K  | train | 0    
1  | bn1       | BatchNorm2d        | 128    | train | 0    
2  | conv2     | Conv2d             | 73.7 K | train | 0    
3  | bn2       | BatchNorm2d        | 256    | train | 0    
4  | conv3     | Conv2d             | 294 K  | train | 0    
5  | bn3       | BatchNorm2d        | 512    | train | 0    
6  | conv4     | Conv2d             | 1.2 M  | train | 0    
7  | bn4       | BatchNorm2d        | 1.0 K  | train | 0    
8  | gap       | AdaptiveAvgPool2d  | 0      | train | 0    
9  | fc1       | Linear             | 131 K  | train | 0    
10 | dropout1  | Dropout            | 0      | train | 0    
11 | fc2       | Linear             | 32.9 K | train | 0    
12 | dropout2  | Dropout            | 0      | train | 0    
13 | fc3       | Linear             | 1.3 K  | train | 0    
14 | criterion | CrossEntropyLoss   | 0      | train | 0    
15 | train_acc | MulticlassAccuracy | 0      | train | 0    
16 | val_acc   | MulticlassAccuracy | 0      | train | 0    
17 | test_acc  | MulticlassAccuracy | 0      | train | 0    
------------------------------------------------------------------
1.7 M     Trainable params
0         Non-trainable params
1.7 M     Total params
6.870     Total estimated model params size (MB)
18        Modules in train mode
0         Modules in eval mode
0         Total Flops
Sanity Checking: |                                                                               | 0/? [00:00<?, ?it/s]E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=17` in the `DataLoader` to improve performance.
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\torch\utils\data\dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=17` in the `DataLoader` to improve performance.
Epoch 0: 100%|█| 782/782 [01:39<00:00,  7.83it/s, v_num=1, train_loss=1.740, train_acc=0.500, val_loss=1.400, val_accurMetric val_accuracy improved. New best score: 0.496                                                                     
Epoch 0, global step 782: 'val_accuracy' reached 0.49560 (best 0.49560), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=00-val_accuracy=0.4956.ckpt' as top 3
Epoch 1: 100%|█| 782/782 [01:44<00:00,  7.45it/s, v_num=1, train_loss=0.852, train_acc=0.688, val_loss=1.180, val_accurMetric val_accuracy improved by 0.094 >= min_delta = 0.001. New best score: 0.590                                       
Epoch 1, global step 1564: 'val_accuracy' reached 0.59000 (best 0.59000), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=01-val_accuracy=0.5900.ckpt' as top 3
Epoch 2: 100%|█| 782/782 [01:44<00:00,  7.50it/s, v_num=1, train_loss=1.150, train_acc=0.562, val_loss=0.926, val_accurMetric val_accuracy improved by 0.082 >= min_delta = 0.001. New best score: 0.672                                       
Epoch 2, global step 2346: 'val_accuracy' reached 0.67190 (best 0.67190), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=02-val_accuracy=0.6719.ckpt' as top 3
Epoch 3: 100%|█| 782/782 [01:46<00:00,  7.31it/s, v_num=1, train_loss=0.863, train_acc=0.750, val_loss=0.901, val_accurMetric val_accuracy improved by 0.014 >= min_delta = 0.001. New best score: 0.685                                       
Epoch 3, global step 3128: 'val_accuracy' reached 0.68540 (best 0.68540), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=03-val_accuracy=0.6854.ckpt' as top 3
Epoch 4: 100%|█| 782/782 [01:39<00:00,  7.87it/s, v_num=1, train_loss=0.929, train_acc=0.562, val_loss=0.796, val_accurMetric val_accuracy improved by 0.041 >= min_delta = 0.001. New best score: 0.726                                       
Epoch 4, global step 3910: 'val_accuracy' reached 0.72620 (best 0.72620), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=04-val_accuracy=0.7262.ckpt' as top 3
Epoch 5: 100%|█| 782/782 [01:38<00:00,  7.94it/s, v_num=1, train_loss=0.865, train_acc=0.750, val_loss=0.791, val_accurMetric val_accuracy improved by 0.007 >= min_delta = 0.001. New best score: 0.733                                       
Epoch 5, global step 4692: 'val_accuracy' reached 0.73290 (best 0.73290), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=05-val_accuracy=0.7329.ckpt' as top 3
Epoch 6: 100%|█| 782/782 [01:41<00:00,  7.71it/s, v_num=1, train_loss=0.706, train_acc=0.750, val_loss=0.689, val_accurMetric val_accuracy improved by 0.028 >= min_delta = 0.001. New best score: 0.761                                       
Epoch 6, global step 5474: 'val_accuracy' reached 0.76120 (best 0.76120), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=06-val_accuracy=0.7612.ckpt' as top 3
Epoch 7: 100%|█| 782/782 [01:38<00:00,  7.93it/s, v_num=1, train_loss=0.651, train_acc=0.688, val_loss=0.693, val_accurEpoch 7, global step 6256: 'val_accuracy' reached 0.76000 (best 0.76120), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=07-val_accuracy=0.7600.ckpt' as top 3
Epoch 8: 100%|█| 782/782 [01:41<00:00,  7.72it/s, v_num=1, train_loss=0.652, train_acc=0.812, val_loss=0.626, val_accurMetric val_accuracy improved by 0.028 >= min_delta = 0.001. New best score: 0.790                                       
Epoch 8, global step 7038: 'val_accuracy' reached 0.78970 (best 0.78970), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=08-val_accuracy=0.7897.ckpt' as top 3
Epoch 9: 100%|█| 782/782 [01:38<00:00,  7.96it/s, v_num=1, train_loss=0.767, train_acc=0.750, val_loss=0.622, val_accurEpoch 9, global step 7820: 'val_accuracy' reached 0.78830 (best 0.78970), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=09-val_accuracy=0.7883.ckpt' as top 3
Epoch 10: 100%|█| 782/782 [01:44<00:00,  7.45it/s, v_num=1, train_loss=1.260, train_acc=0.688, val_loss=0.592, val_accuMetric val_accuracy improved by 0.012 >= min_delta = 0.001. New best score: 0.801                                       
Epoch 10, global step 8602: 'val_accuracy' reached 0.80140 (best 0.80140), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=10-val_accuracy=0.8014.ckpt' as top 3
Epoch 11: 100%|█| 782/782 [01:44<00:00,  7.51it/s, v_num=1, train_loss=0.616, train_acc=0.688, val_loss=0.578, val_accuMetric val_accuracy improved by 0.003 >= min_delta = 0.001. New best score: 0.804                                       
Epoch 11, global step 9384: 'val_accuracy' reached 0.80400 (best 0.80400), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=11-val_accuracy=0.8040.ckpt' as top 3
Epoch 12: 100%|█| 782/782 [01:43<00:00,  7.57it/s, v_num=1, train_loss=0.880, train_acc=0.625, val_loss=0.567, val_accuMetric val_accuracy improved by 0.011 >= min_delta = 0.001. New best score: 0.815                                       
Epoch 12, global step 10166: 'val_accuracy' reached 0.81480 (best 0.81480), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=12-val_accuracy=0.8148.ckpt' as top 3
Epoch 13: 100%|█| 782/782 [01:40<00:00,  7.76it/s, v_num=1, train_loss=1.000, train_acc=0.750, val_loss=0.524, val_accuMetric val_accuracy improved by 0.009 >= min_delta = 0.001. New best score: 0.823                                       
Epoch 13, global step 10948: 'val_accuracy' reached 0.82340 (best 0.82340), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=13-val_accuracy=0.8234.ckpt' as top 3
Epoch 14: 100%|█| 782/782 [01:42<00:00,  7.62it/s, v_num=1, train_loss=0.617, train_acc=0.750, val_loss=0.512, val_accuMetric val_accuracy improved by 0.003 >= min_delta = 0.001. New best score: 0.826                                       
Epoch 14, global step 11730: 'val_accuracy' reached 0.82600 (best 0.82600), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=14-val_accuracy=0.8260.ckpt' as top 3
Epoch 15: 100%|█| 782/782 [01:49<00:00,  7.16it/s, v_num=1, train_loss=0.676, train_acc=0.875, val_loss=0.500, val_accuMetric val_accuracy improved by 0.006 >= min_delta = 0.001. New best score: 0.832                                       
Epoch 15, global step 12512: 'val_accuracy' reached 0.83190 (best 0.83190), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=15-val_accuracy=0.8319.ckpt' as top 3
Epoch 16: 100%|█| 782/782 [01:46<00:00,  7.36it/s, v_num=1, train_loss=0.325, train_acc=0.875, val_loss=0.469, val_accuMetric val_accuracy improved by 0.008 >= min_delta = 0.001. New best score: 0.840                                       
Epoch 16, global step 13294: 'val_accuracy' reached 0.83950 (best 0.83950), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=16-val_accuracy=0.8395.ckpt' as top 3
Epoch 17: 100%|█| 782/782 [01:42<00:00,  7.63it/s, v_num=1, train_loss=1.080, train_acc=0.625, val_loss=0.467, val_accuMetric val_accuracy improved by 0.003 >= min_delta = 0.001. New best score: 0.843                                       
Epoch 17, global step 14076: 'val_accuracy' reached 0.84300 (best 0.84300), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=17-val_accuracy=0.8430.ckpt' as top 3
Epoch 18: 100%|█| 782/782 [01:37<00:00,  8.03it/s, v_num=1, train_loss=0.334, train_acc=0.938, val_loss=0.460, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.001. New best score: 0.845                                       
Epoch 18, global step 14858: 'val_accuracy' reached 0.84550 (best 0.84550), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=18-val_accuracy=0.8455.ckpt' as top 3
Epoch 19: 100%|█| 782/782 [01:39<00:00,  7.86it/s, v_num=1, train_loss=0.683, train_acc=0.812, val_loss=0.443, val_accuMetric val_accuracy improved by 0.006 >= min_delta = 0.001. New best score: 0.852                                       
Epoch 19, global step 15640: 'val_accuracy' reached 0.85160 (best 0.85160), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=19-val_accuracy=0.8516.ckpt' as top 3
Epoch 20: 100%|█| 782/782 [01:37<00:00,  8.05it/s, v_num=1, train_loss=0.347, train_acc=0.812, val_loss=0.444, val_accuMetric val_accuracy improved by 0.004 >= min_delta = 0.001. New best score: 0.855                                       
Epoch 20, global step 16422: 'val_accuracy' reached 0.85530 (best 0.85530), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=20-val_accuracy=0.8553.ckpt' as top 3
Epoch 21: 100%|█| 782/782 [01:42<00:00,  7.64it/s, v_num=1, train_loss=0.381, train_acc=0.750, val_loss=0.426, val_accuMetric val_accuracy improved by 0.005 >= min_delta = 0.001. New best score: 0.860                                       
Epoch 21, global step 17204: 'val_accuracy' reached 0.86000 (best 0.86000), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=21-val_accuracy=0.8600.ckpt' as top 3
Epoch 22: 100%|█| 782/782 [01:48<00:00,  7.21it/s, v_num=1, train_loss=0.307, train_acc=0.875, val_loss=0.424, val_accuEpoch 22, global step 17986: 'val_accuracy' reached 0.86050 (best 0.86050), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=22-val_accuracy=0.8605.ckpt' as top 3
Epoch 23: 100%|█| 782/782 [01:37<00:00,  8.02it/s, v_num=1, train_loss=0.519, train_acc=0.750, val_loss=0.427, val_accuEpoch 23, global step 18768: 'val_accuracy' reached 0.85830 (best 0.86050), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=23-val_accuracy=0.8583.ckpt' as top 3
Epoch 24: 100%|█| 782/782 [01:37<00:00,  8.02it/s, v_num=1, train_loss=0.549, train_acc=0.875, val_loss=0.413, val_accuMetric val_accuracy improved by 0.005 >= min_delta = 0.001. New best score: 0.865                                       
Epoch 24, global step 19550: 'val_accuracy' reached 0.86470 (best 0.86470), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=24-val_accuracy=0.8647.ckpt' as top 3
Epoch 25: 100%|█| 782/782 [01:39<00:00,  7.83it/s, v_num=1, train_loss=0.769, train_acc=0.875, val_loss=0.413, val_accuEpoch 25, global step 20332: 'val_accuracy' reached 0.86290 (best 0.86470), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=25-val_accuracy=0.8629.ckpt' as top 3
Epoch 26: 100%|█| 782/782 [01:42<00:00,  7.67it/s, v_num=1, train_loss=0.110, train_acc=0.938, val_loss=0.412, val_accuEpoch 26, global step 21114: 'val_accuracy' reached 0.86290 (best 0.86470), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=26-val_accuracy=0.8629.ckpt' as top 3
Epoch 27: 100%|█| 782/782 [01:42<00:00,  7.59it/s, v_num=1, train_loss=0.435, train_acc=0.812, val_loss=0.409, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.001. New best score: 0.866                                       
Epoch 27, global step 21896: 'val_accuracy' reached 0.86620 (best 0.86620), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=27-val_accuracy=0.8662.ckpt' as top 3
Epoch 28: 100%|█| 782/782 [01:44<00:00,  7.50it/s, v_num=1, train_loss=0.856, train_acc=0.688, val_loss=0.408, val_accuEpoch 28, global step 22678: 'val_accuracy' reached 0.86510 (best 0.86620), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=28-val_accuracy=0.8651.ckpt' as top 3
Epoch 29: 100%|█| 782/782 [01:37<00:00,  8.00it/s, v_num=1, train_loss=0.283, train_acc=0.938, val_loss=0.408, val_accuEpoch 29, global step 23460: 'val_accuracy' was not in top 3                                                            
`Trainer.fit` stopped: `max_epochs=30` reached.
Epoch 29: 100%|█| 782/782 [01:37<00:00,  8.00it/s, v_num=1, train_loss=0.283, train_acc=0.938, val_loss=0.408, val_accu

Testing best model...
Loading best model: E:\python_exercises\zms_cifar10_cnn\checkpoints\cifar10-cnn-epoch=27-val_accuracy=0.8662.ckpt
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=17` in the `DataLoader` to improve performance.
Testing DataLoader 0: 100%|██████████████████████████████████████████████████████████| 157/157 [00:06<00:00, 26.01it/s]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      test_accuracy         0.8661999702453613
        test_loss           0.40934792160987854
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
============================================================
Final Test Accuracy: 0.8662
Target not reached, but close to 86.62%
============================================================
Model saved to: ./checkpoints\final_model.pth

Training completed!
```

## 实验3：进一步完整测试
- 时间：2026年1月15日
- 配置：batch_size=64, max_epochs=80
- 结果：测试准确率 89.38%
- 性能分析  
起始准确率：51.70%（Epoch 0）  
总提升：37.68个百分点
- 命令行输出：
```bash
(.venv) PS E:\python_exercises\zms_cifar10_cnn> python train.py
============================================================
CIFAR10 CNN Training - Target: 93%+ Accuracy
============================================================
GPU Not Available, using CPU

Loading CIFAR10 dataset...
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\torchvision\datasets\cifar.py:83: VisibleDeprecationWarning: dtype(): align should be passed as Python or NumPy boolean but got `align=0`. Did you mean to pass a tuple to create a subarray type? (Deprecated NumPy 2.4)
  entry = pickle.load(f, encoding="latin1")
Initializing CNN model...
Model Parameters: Total=1,717,450, Trainable=1,717,450
GPU available: False, used: False
TPU available: False, using: 0 TPU cores

Starting training...
Max Epochs: 80
Target Accuracy: 93%+
============================================================
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:881: Checkpoint directory E:\python_exercises\zms_cifar10_cnn\checkpoints exists and is not empty.

   | Name      | Type               | Params | Mode  | FLOPs
------------------------------------------------------------------
0  | conv1     | Conv2d             | 1.7 K  | train | 0    
1  | bn1       | BatchNorm2d        | 128    | train | 0    
2  | conv2     | Conv2d             | 73.7 K | train | 0    
3  | bn2       | BatchNorm2d        | 256    | train | 0    
4  | conv3     | Conv2d             | 294 K  | train | 0    
5  | bn3       | BatchNorm2d        | 512    | train | 0    
6  | conv4     | Conv2d             | 1.2 M  | train | 0    
7  | bn4       | BatchNorm2d        | 1.0 K  | train | 0    
8  | gap       | AdaptiveAvgPool2d  | 0      | train | 0    
9  | fc1       | Linear             | 131 K  | train | 0    
10 | dropout1  | Dropout            | 0      | train | 0    
11 | fc2       | Linear             | 32.9 K | train | 0    
12 | dropout2  | Dropout            | 0      | train | 0    
13 | fc3       | Linear             | 1.3 K  | train | 0    
14 | criterion | CrossEntropyLoss   | 0      | train | 0    
15 | train_acc | MulticlassAccuracy | 0      | train | 0    
16 | val_acc   | MulticlassAccuracy | 0      | train | 0    
17 | test_acc  | MulticlassAccuracy | 0      | train | 0    
------------------------------------------------------------------
1.7 M     Trainable params
0         Non-trainable params
1.7 M     Total params
6.870     Total estimated model params size (MB)
18        Modules in train mode
0         Modules in eval mode
0         Total Flops
Sanity Checking: |                                                                               | 0/? [00:00<?, ?it/s]E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=17` in the `DataLoader` to improve performance.
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\torch\utils\data\dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=17` in the `DataLoader` to improve performance.
Epoch 0: 100%|█| 782/782 [01:35<00:00,  8.17it/s, v_num=2, train_loss=1.610, train_acc=0.438, val_loss=1.370, val_accurMetric val_accuracy improved. New best score: 0.517                                                                     
Epoch 0, global step 782: 'val_accuracy' reached 0.51720 (best 0.51720), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=00-val_accuracy=0.5172.ckpt' as top 3
Epoch 1: 100%|█| 782/782 [01:38<00:00,  7.94it/s, v_num=2, train_loss=0.704, train_acc=0.750, val_loss=1.080, val_accurMetric val_accuracy improved by 0.101 >= min_delta = 0.0005. New best score: 0.618                                      
Epoch 1, global step 1564: 'val_accuracy' reached 0.61850 (best 0.61850), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=01-val_accuracy=0.6185.ckpt' as top 3
Epoch 2: 100%|█| 782/782 [01:38<00:00,  7.96it/s, v_num=2, train_loss=0.776, train_acc=0.688, val_loss=0.987, val_accurMetric val_accuracy improved by 0.043 >= min_delta = 0.0005. New best score: 0.661                                      
Epoch 2, global step 2346: 'val_accuracy' reached 0.66130 (best 0.66130), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=02-val_accuracy=0.6613.ckpt' as top 3
Epoch 3: 100%|█| 782/782 [01:40<00:00,  7.76it/s, v_num=2, train_loss=1.070, train_acc=0.688, val_loss=0.878, val_accurMetric val_accuracy improved by 0.031 >= min_delta = 0.0005. New best score: 0.693                                      
Epoch 3, global step 3128: 'val_accuracy' reached 0.69270 (best 0.69270), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=03-val_accuracy=0.6927.ckpt' as top 3
Epoch 4: 100%|█| 782/782 [01:39<00:00,  7.82it/s, v_num=2, train_loss=1.120, train_acc=0.562, val_loss=0.777, val_accurMetric val_accuracy improved by 0.032 >= min_delta = 0.0005. New best score: 0.724                                      
Epoch 4, global step 3910: 'val_accuracy' reached 0.72440 (best 0.72440), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=04-val_accuracy=0.7244.ckpt' as top 3
Epoch 5: 100%|█| 782/782 [01:39<00:00,  7.86it/s, v_num=2, train_loss=1.140, train_acc=0.562, val_loss=0.819, val_accurEpoch 5, global step 4692: 'val_accuracy' reached 0.71880 (best 0.72440), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=05-val_accuracy=0.7188.ckpt' as top 3
Epoch 6: 100%|█| 782/782 [01:40<00:00,  7.77it/s, v_num=2, train_loss=1.280, train_acc=0.562, val_loss=0.701, val_accurMetric val_accuracy improved by 0.039 >= min_delta = 0.0005. New best score: 0.763                                      
Epoch 6, global step 5474: 'val_accuracy' reached 0.76320 (best 0.76320), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=06-val_accuracy=0.7632.ckpt' as top 3
Epoch 7: 100%|█| 782/782 [01:46<00:00,  7.37it/s, v_num=2, train_loss=0.765, train_acc=0.625, val_loss=0.644, val_accurMetric val_accuracy improved by 0.019 >= min_delta = 0.0005. New best score: 0.782                                      
Epoch 7, global step 6256: 'val_accuracy' reached 0.78240 (best 0.78240), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=07-val_accuracy=0.7824.ckpt' as top 3
Epoch 8: 100%|█| 782/782 [01:53<00:00,  6.87it/s, v_num=2, train_loss=0.890, train_acc=0.688, val_loss=0.658, val_accurEpoch 8, global step 7038: 'val_accuracy' reached 0.77430 (best 0.78240), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=08-val_accuracy=0.7743.ckpt' as top 3
Epoch 9: 100%|█| 782/782 [01:50<00:00,  7.05it/s, v_num=2, train_loss=0.856, train_acc=0.688, val_loss=0.593, val_accurMetric val_accuracy improved by 0.017 >= min_delta = 0.0005. New best score: 0.799                                      
Epoch 9, global step 7820: 'val_accuracy' reached 0.79940 (best 0.79940), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=09-val_accuracy=0.7994.ckpt' as top 3
Epoch 10: 100%|█| 782/782 [01:42<00:00,  7.63it/s, v_num=2, train_loss=1.150, train_acc=0.625, val_loss=0.584, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.801                                      
Epoch 10, global step 8602: 'val_accuracy' reached 0.80130 (best 0.80130), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=10-val_accuracy=0.8013.ckpt' as top 3
Epoch 11: 100%|█| 782/782 [01:41<00:00,  7.67it/s, v_num=2, train_loss=0.578, train_acc=0.812, val_loss=0.560, val_accuMetric val_accuracy improved by 0.010 >= min_delta = 0.0005. New best score: 0.811                                      
Epoch 11, global step 9384: 'val_accuracy' reached 0.81110 (best 0.81110), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=11-val_accuracy=0.8111.ckpt' as top 3
Epoch 12: 100%|█| 782/782 [01:39<00:00,  7.83it/s, v_num=2, train_loss=0.604, train_acc=0.750, val_loss=0.575, val_accuEpoch 12, global step 10166: 'val_accuracy' reached 0.80500 (best 0.81110), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=12-val_accuracy=0.8050.ckpt' as top 3
Epoch 13: 100%|█| 782/782 [01:38<00:00,  7.96it/s, v_num=2, train_loss=0.479, train_acc=0.812, val_loss=0.600, val_accuEpoch 13, global step 10948: 'val_accuracy' was not in top 3                                                            
Epoch 14: 100%|█| 782/782 [01:38<00:00,  7.92it/s, v_num=2, train_loss=0.592, train_acc=0.812, val_loss=0.637, val_accuEpoch 14, global step 11730: 'val_accuracy' was not in top 3                                                            
Epoch 15: 100%|█| 782/782 [07:14<00:00,  1.80it/s, v_num=2, train_loss=0.800, train_acc=0.688, val_loss=0.508, val_accuMetric val_accuracy improved by 0.013 >= min_delta = 0.0005. New best score: 0.824                                      
Epoch 15, global step 12512: 'val_accuracy' reached 0.82430 (best 0.82430), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=15-val_accuracy=0.8243.ckpt' as top 3
Epoch 16: 100%|█| 782/782 [01:38<00:00,  7.91it/s, v_num=2, train_loss=0.719, train_acc=0.812, val_loss=0.513, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.826                                      
Epoch 16, global step 13294: 'val_accuracy' reached 0.82610 (best 0.82610), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=16-val_accuracy=0.8261.ckpt' as top 3
Epoch 17: 100%|█| 782/782 [01:44<00:00,  7.46it/s, v_num=2, train_loss=1.280, train_acc=0.688, val_loss=0.499, val_accuMetric val_accuracy improved by 0.007 >= min_delta = 0.0005. New best score: 0.833                                      
Epoch 17, global step 14076: 'val_accuracy' reached 0.83260 (best 0.83260), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=17-val_accuracy=0.8326.ckpt' as top 3
Epoch 18: 100%|█| 782/782 [01:48<00:00,  7.20it/s, v_num=2, train_loss=0.547, train_acc=0.750, val_loss=0.471, val_accuMetric val_accuracy improved by 0.008 >= min_delta = 0.0005. New best score: 0.841                                      
Epoch 18, global step 14858: 'val_accuracy' reached 0.84080 (best 0.84080), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=18-val_accuracy=0.8408.ckpt' as top 3
Epoch 19: 100%|█| 782/782 [04:38<00:00,  2.81it/s, v_num=2, train_loss=0.234, train_acc=1.000, val_loss=0.459, val_accuMetric val_accuracy improved by 0.006 >= min_delta = 0.0005. New best score: 0.847                                      
Epoch 19, global step 15640: 'val_accuracy' reached 0.84690 (best 0.84690), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=19-val_accuracy=0.8469.ckpt' as top 3
Epoch 20: 100%|█| 782/782 [01:42<00:00,  7.61it/s, v_num=2, train_loss=0.421, train_acc=0.875, val_loss=0.455, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.848                                      
Epoch 20, global step 16422: 'val_accuracy' reached 0.84750 (best 0.84750), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=20-val_accuracy=0.8475.ckpt' as top 3
Epoch 21: 100%|█| 782/782 [01:38<00:00,  7.94it/s, v_num=2, train_loss=0.126, train_acc=1.000, val_loss=0.464, val_accuEpoch 21, global step 17204: 'val_accuracy' reached 0.84320 (best 0.84750), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=21-val_accuracy=0.8432.ckpt' as top 3
Epoch 22: 100%|█| 782/782 [01:39<00:00,  7.89it/s, v_num=2, train_loss=0.444, train_acc=0.875, val_loss=0.440, val_accuMetric val_accuracy improved by 0.009 >= min_delta = 0.0005. New best score: 0.856                                      
Epoch 22, global step 17986: 'val_accuracy' reached 0.85640 (best 0.85640), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=22-val_accuracy=0.8564.ckpt' as top 3
Epoch 23: 100%|█| 782/782 [01:44<00:00,  7.48it/s, v_num=2, train_loss=0.671, train_acc=0.875, val_loss=0.462, val_accuEpoch 23, global step 18768: 'val_accuracy' was not in top 3                                                            
Epoch 24: 100%|█| 782/782 [03:31<00:00,  3.70it/s, v_num=2, train_loss=0.410, train_acc=0.938, val_loss=0.424, val_accuMetric val_accuracy improved by 0.003 >= min_delta = 0.0005. New best score: 0.859                                      
Epoch 24, global step 19550: 'val_accuracy' reached 0.85900 (best 0.85900), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=24-val_accuracy=0.8590.ckpt' as top 3
Epoch 25: 100%|█| 782/782 [01:38<00:00,  7.91it/s, v_num=2, train_loss=0.338, train_acc=0.875, val_loss=0.446, val_accuEpoch 25, global step 20332: 'val_accuracy' reached 0.85200 (best 0.85900), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=25-val_accuracy=0.8520.ckpt' as top 3
Epoch 26: 100%|█| 782/782 [01:39<00:00,  7.90it/s, v_num=2, train_loss=0.377, train_acc=0.875, val_loss=0.447, val_accuEpoch 26, global step 21114: 'val_accuracy' was not in top 3                                                            
Epoch 27: 100%|█| 782/782 [01:38<00:00,  7.93it/s, v_num=2, train_loss=0.608, train_acc=0.688, val_loss=0.434, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.860                                      
Epoch 27, global step 21896: 'val_accuracy' reached 0.86040 (best 0.86040), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=27-val_accuracy=0.8604.ckpt' as top 3
Epoch 28: 100%|█| 782/782 [01:39<00:00,  7.86it/s, v_num=2, train_loss=0.795, train_acc=0.812, val_loss=0.422, val_accuMetric val_accuracy improved by 0.004 >= min_delta = 0.0005. New best score: 0.864                                      
Epoch 28, global step 22678: 'val_accuracy' reached 0.86450 (best 0.86450), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=28-val_accuracy=0.8645.ckpt' as top 3
Epoch 29: 100%|█| 782/782 [01:40<00:00,  7.80it/s, v_num=2, train_loss=0.261, train_acc=0.938, val_loss=0.406, val_accuMetric val_accuracy improved by 0.004 >= min_delta = 0.0005. New best score: 0.868                                      
Epoch 29, global step 23460: 'val_accuracy' reached 0.86820 (best 0.86820), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=29-val_accuracy=0.8682.ckpt' as top 3
Epoch 30: 100%|█| 782/782 [03:47<00:00,  3.44it/s, v_num=2, train_loss=0.913, train_acc=0.500, val_loss=0.418, val_accuEpoch 30, global step 24242: 'val_accuracy' reached 0.86460 (best 0.86820), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=30-val_accuracy=0.8646.ckpt' as top 3
Epoch 31: 100%|█| 782/782 [01:37<00:00,  7.99it/s, v_num=2, train_loss=0.139, train_acc=0.938, val_loss=0.407, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.869                                      
Epoch 31, global step 25024: 'val_accuracy' reached 0.86940 (best 0.86940), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=31-val_accuracy=0.8694.ckpt' as top 3
Epoch 32: 100%|█| 782/782 [01:37<00:00,  8.01it/s, v_num=2, train_loss=0.510, train_acc=0.812, val_loss=0.420, val_accuEpoch 32, global step 25806: 'val_accuracy' reached 0.86510 (best 0.86940), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=32-val_accuracy=0.8651.ckpt' as top 3
Epoch 33: 100%|█| 782/782 [01:39<00:00,  7.90it/s, v_num=2, train_loss=0.336, train_acc=0.875, val_loss=0.406, val_accuEpoch 33, global step 26588: 'val_accuracy' reached 0.86810 (best 0.86940), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=33-val_accuracy=0.8681.ckpt' as top 3
Epoch 34: 100%|█| 782/782 [01:38<00:00,  7.94it/s, v_num=2, train_loss=0.206, train_acc=0.938, val_loss=0.419, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.871                                      
Epoch 34, global step 27370: 'val_accuracy' reached 0.87070 (best 0.87070), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=34-val_accuracy=0.8707.ckpt' as top 3
Epoch 35: 100%|█| 782/782 [09:58<00:00,  1.31it/s, v_num=2, train_loss=0.266, train_acc=0.938, val_loss=0.415, val_accuMetric val_accuracy improved by 0.004 >= min_delta = 0.0005. New best score: 0.874                                      
Epoch 35, global step 28152: 'val_accuracy' reached 0.87450 (best 0.87450), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=35-val_accuracy=0.8745.ckpt' as top 3
Epoch 36: 100%|█| 782/782 [01:39<00:00,  7.84it/s, v_num=2, train_loss=0.0625, train_acc=1.000, val_loss=0.396, val_accMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.876                                      
Epoch 36, global step 28934: 'val_accuracy' reached 0.87640 (best 0.87640), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=36-val_accuracy=0.8764.ckpt' as top 3
Epoch 37: 100%|█| 782/782 [01:37<00:00,  7.98it/s, v_num=2, train_loss=0.0664, train_acc=1.000, val_loss=0.404, val_accEpoch 37, global step 29716: 'val_accuracy' was not in top 3                                                            
Epoch 38: 100%|█| 782/782 [01:37<00:00,  8.02it/s, v_num=2, train_loss=1.290, train_acc=0.750, val_loss=0.404, val_accuEpoch 38, global step 30498: 'val_accuracy' reached 0.87450 (best 0.87640), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=38-val_accuracy=0.8745.ckpt' as top 3
Epoch 39: 100%|█| 782/782 [01:38<00:00,  7.92it/s, v_num=2, train_loss=0.363, train_acc=0.812, val_loss=0.402, val_accuEpoch 39, global step 31280: 'val_accuracy' reached 0.87670 (best 0.87670), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=39-val_accuracy=0.8767.ckpt' as top 3
Epoch 40: 100%|█| 782/782 [01:39<00:00,  7.89it/s, v_num=2, train_loss=0.290, train_acc=0.875, val_loss=0.394, val_accuMetric val_accuracy improved by 0.004 >= min_delta = 0.0005. New best score: 0.881                                      
Epoch 40, global step 32062: 'val_accuracy' reached 0.88060 (best 0.88060), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=40-val_accuracy=0.8806.ckpt' as top 3
Epoch 41: 100%|█| 782/782 [01:40<00:00,  7.77it/s, v_num=2, train_loss=0.238, train_acc=0.938, val_loss=0.384, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.883                                      
Epoch 41, global step 32844: 'val_accuracy' reached 0.88310 (best 0.88310), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=41-val_accuracy=0.8831.ckpt' as top 3
Epoch 42: 100%|█| 782/782 [01:38<00:00,  7.94it/s, v_num=2, train_loss=0.635, train_acc=0.875, val_loss=0.390, val_accuEpoch 42, global step 33626: 'val_accuracy' reached 0.88030 (best 0.88310), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=42-val_accuracy=0.8803.ckpt' as top 3
Epoch 43: 100%|█| 782/782 [01:40<00:00,  7.78it/s, v_num=2, train_loss=0.542, train_acc=0.812, val_loss=0.394, val_accuEpoch 43, global step 34408: 'val_accuracy' was not in top 3                                                            
Epoch 44: 100%|█| 782/782 [01:58<00:00,  6.62it/s, v_num=2, train_loss=0.391, train_acc=0.938, val_loss=0.404, val_accuEpoch 44, global step 35190: 'val_accuracy' reached 0.88080 (best 0.88310), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=44-val_accuracy=0.8808.ckpt' as top 3
Epoch 45: 100%|█| 782/782 [01:41<00:00,  7.68it/s, v_num=2, train_loss=0.0937, train_acc=0.938, val_loss=0.379, val_accMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.885                                      
Epoch 45, global step 35972: 'val_accuracy' reached 0.88480 (best 0.88480), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=45-val_accuracy=0.8848.ckpt' as top 3
Epoch 46: 100%|█| 782/782 [01:44<00:00,  7.49it/s, v_num=2, train_loss=0.559, train_acc=0.812, val_loss=0.389, val_accuEpoch 46, global step 36754: 'val_accuracy' reached 0.88120 (best 0.88480), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=46-val_accuracy=0.8812.ckpt' as top 3
Epoch 47: 100%|█| 782/782 [01:37<00:00,  8.03it/s, v_num=2, train_loss=0.139, train_acc=1.000, val_loss=0.384, val_accuEpoch 47, global step 37536: 'val_accuracy' reached 0.88480 (best 0.88480), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=47-val_accuracy=0.8848.ckpt' as top 3
Epoch 48: 100%|█| 782/782 [01:36<00:00,  8.07it/s, v_num=2, train_loss=0.0998, train_acc=0.938, val_loss=0.398, val_accEpoch 48, global step 38318: 'val_accuracy' reached 0.88380 (best 0.88480), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=48-val_accuracy=0.8838.ckpt' as top 3
Epoch 49: 100%|█| 782/782 [01:36<00:00,  8.13it/s, v_num=2, train_loss=0.407, train_acc=0.875, val_loss=0.387, val_accuEpoch 49, global step 39100: 'val_accuracy' was not in top 3                                                            
Epoch 50: 100%|█| 782/782 [01:35<00:00,  8.19it/s, v_num=2, train_loss=0.107, train_acc=1.000, val_loss=0.393, val_accuEpoch 50, global step 39882: 'val_accuracy' reached 0.88500 (best 0.88500), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=50-val_accuracy=0.8850.ckpt' as top 3
Epoch 51: 100%|█| 782/782 [01:35<00:00,  8.19it/s, v_num=2, train_loss=0.106, train_acc=1.000, val_loss=0.388, val_accuEpoch 51, global step 40664: 'val_accuracy' was not in top 3                                                            
Epoch 52: 100%|█| 782/782 [01:35<00:00,  8.20it/s, v_num=2, train_loss=0.216, train_acc=0.875, val_loss=0.388, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.886                                      
Epoch 52, global step 41446: 'val_accuracy' reached 0.88600 (best 0.88600), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=52-val_accuracy=0.8860.ckpt' as top 3
Epoch 53: 100%|█| 782/782 [01:37<00:00,  8.06it/s, v_num=2, train_loss=0.167, train_acc=0.938, val_loss=0.381, val_accuEpoch 53, global step 42228: 'val_accuracy' reached 0.88570 (best 0.88600), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=53-val_accuracy=0.8857.ckpt' as top 3
Epoch 54: 100%|█| 782/782 [01:35<00:00,  8.15it/s, v_num=2, train_loss=0.230, train_acc=0.938, val_loss=0.386, val_accuMetric val_accuracy improved by 0.005 >= min_delta = 0.0005. New best score: 0.891                                      
Epoch 54, global step 43010: 'val_accuracy' reached 0.89120 (best 0.89120), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=54-val_accuracy=0.8912.ckpt' as top 3
Epoch 55: 100%|█| 782/782 [01:36<00:00,  8.09it/s, v_num=2, train_loss=0.241, train_acc=0.875, val_loss=0.385, val_accuEpoch 55, global step 43792: 'val_accuracy' reached 0.89120 (best 0.89120), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=55-val_accuracy=0.8912.ckpt' as top 3
Epoch 56: 100%|█| 782/782 [01:38<00:00,  7.93it/s, v_num=2, train_loss=0.642, train_acc=0.812, val_loss=0.385, val_accuEpoch 56, global step 44574: 'val_accuracy' reached 0.88860 (best 0.89120), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=56-val_accuracy=0.8886.ckpt' as top 3
Epoch 57: 100%|█| 782/782 [01:35<00:00,  8.16it/s, v_num=2, train_loss=0.218, train_acc=0.938, val_loss=0.387, val_accuEpoch 57, global step 45356: 'val_accuracy' was not in top 3                                                            
Epoch 58: 100%|█| 782/782 [01:36<00:00,  8.13it/s, v_num=2, train_loss=0.142, train_acc=0.938, val_loss=0.387, val_accuEpoch 58, global step 46138: 'val_accuracy' was not in top 3                                                            
Epoch 59: 100%|█| 782/782 [01:36<00:00,  8.12it/s, v_num=2, train_loss=0.333, train_acc=0.875, val_loss=0.381, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.892                                      
Epoch 59, global step 46920: 'val_accuracy' reached 0.89210 (best 0.89210), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=59-val_accuracy=0.8921.ckpt' as top 3
Epoch 60: 100%|█| 782/782 [01:39<00:00,  7.86it/s, v_num=2, train_loss=0.174, train_acc=0.938, val_loss=0.387, val_accuEpoch 60, global step 47702: 'val_accuracy' was not in top 3                                                            
Epoch 61: 100%|█| 782/782 [01:38<00:00,  7.95it/s, v_num=2, train_loss=0.668, train_acc=0.875, val_loss=0.391, val_accuEpoch 61, global step 48484: 'val_accuracy' was not in top 3                                                            
Epoch 62: 100%|█| 782/782 [01:35<00:00,  8.15it/s, v_num=2, train_loss=0.0936, train_acc=0.938, val_loss=0.396, val_accEpoch 62, global step 49266: 'val_accuracy' reached 0.89140 (best 0.89210), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=62-val_accuracy=0.8914.ckpt' as top 3
Epoch 63: 100%|█| 782/782 [01:37<00:00,  8.03it/s, v_num=2, train_loss=0.0362, train_acc=1.000, val_loss=0.386, val_accEpoch 63, global step 50048: 'val_accuracy' reached 0.89250 (best 0.89250), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=63-val_accuracy=0.8925.ckpt' as top 3
Epoch 64: 100%|█| 782/782 [04:02<00:00,  3.22it/s, v_num=2, train_loss=0.284, train_acc=0.875, val_loss=0.395, val_accuEpoch 64, global step 50830: 'val_accuracy' was not in top 3                                                            
Epoch 65: 100%|█| 782/782 [01:39<00:00,  7.87it/s, v_num=2, train_loss=0.0441, train_acc=1.000, val_loss=0.391, val_accEpoch 65, global step 51612: 'val_accuracy' was not in top 3                                                            
Epoch 66: 100%|█| 782/782 [01:39<00:00,  7.87it/s, v_num=2, train_loss=0.124, train_acc=0.938, val_loss=0.391, val_accuEpoch 66, global step 52394: 'val_accuracy' was not in top 3                                                            
Epoch 67: 100%|█| 782/782 [01:38<00:00,  7.92it/s, v_num=2, train_loss=0.0521, train_acc=1.000, val_loss=0.393, val_accEpoch 67, global step 53176: 'val_accuracy' reached 0.89210 (best 0.89250), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=67-val_accuracy=0.8921.ckpt' as top 3
Epoch 68: 100%|█| 782/782 [01:38<00:00,  7.94it/s, v_num=2, train_loss=0.413, train_acc=0.875, val_loss=0.393, val_accuEpoch 68, global step 53958: 'val_accuracy' reached 0.89220 (best 0.89250), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=68-val_accuracy=0.8922.ckpt' as top 3
Epoch 69: 100%|█| 782/782 [01:38<00:00,  7.97it/s, v_num=2, train_loss=0.700, train_acc=0.812, val_loss=0.396, val_accuEpoch 69, global step 54740: 'val_accuracy' was not in top 3                                                            
Epoch 70: 100%|█| 782/782 [01:38<00:00,  7.91it/s, v_num=2, train_loss=0.359, train_acc=0.875, val_loss=0.400, val_accuEpoch 70, global step 55522: 'val_accuracy' was not in top 3                                                            
Epoch 71: 100%|█| 782/782 [01:41<00:00,  7.72it/s, v_num=2, train_loss=0.153, train_acc=0.875, val_loss=0.397, val_accuEpoch 71, global step 56304: 'val_accuracy' was not in top 3                                                            
Epoch 72: 100%|█| 782/782 [33:21<00:00,  0.39it/s, v_num=2, train_loss=0.134, train_acc=0.938, val_loss=0.394, val_accuEpoch 72, global step 57086: 'val_accuracy' was not in top 3                                                            
Epoch 73: 100%|█| 782/782 [01:41<00:00,  7.71it/s, v_num=2, train_loss=0.159, train_acc=0.938, val_loss=0.396, val_accuEpoch 73, global step 57868: 'val_accuracy' reached 0.89250 (best 0.89250), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=73-val_accuracy=0.8925.ckpt' as top 3
Epoch 74: 100%|█| 782/782 [01:40<00:00,  7.75it/s, v_num=2, train_loss=0.412, train_acc=0.938, val_loss=0.392, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.894                                      
Epoch 74, global step 58650: 'val_accuracy' reached 0.89380 (best 0.89380), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=74-val_accuracy=0.8938.ckpt' as top 3
Epoch 75: 100%|█| 782/782 [01:41<00:00,  7.69it/s, v_num=2, train_loss=0.108, train_acc=1.000, val_loss=0.393, val_accuEpoch 75, global step 59432: 'val_accuracy' was not in top 3                                                            
Epoch 76: 100%|█| 782/782 [01:37<00:00,  7.98it/s, v_num=2, train_loss=0.407, train_acc=0.938, val_loss=0.396, val_accuEpoch 76, global step 60214: 'val_accuracy' reached 0.89290 (best 0.89380), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=76-val_accuracy=0.8929.ckpt' as top 3
Epoch 77: 100%|█| 782/782 [01:37<00:00,  8.01it/s, v_num=2, train_loss=0.0815, train_acc=0.938, val_loss=0.396, val_accEpoch 77, global step 60996: 'val_accuracy' reached 0.89360 (best 0.89380), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=77-val_accuracy=0.8936.ckpt' as top 3
Epoch 78: 100%|█| 782/782 [01:36<00:00,  8.09it/s, v_num=2, train_loss=0.580, train_acc=0.875, val_loss=0.388, val_accuEpoch 78, global step 61778: 'val_accuracy' reached 0.89330 (best 0.89380), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=78-val_accuracy=0.8933.ckpt' as top 3
Epoch 79: 100%|█| 782/782 [01:37<00:00,  8.05it/s, v_num=2, train_loss=0.403, train_acc=0.812, val_loss=0.397, val_accuEpoch 79, global step 62560: 'val_accuracy' was not in top 3                                                            
`Trainer.fit` stopped: `max_epochs=80` reached.
Epoch 79: 100%|█| 782/782 [01:37<00:00,  8.05it/s, v_num=2, train_loss=0.403, train_acc=0.812, val_loss=0.397, val_accu

Testing best model...
Loading best model: E:\python_exercises\zms_cifar10_cnn\checkpoints\cifar10-cnn-epoch=74-val_accuracy=0.8938.ckpt
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=17` in the `DataLoader` to improve performance.
Testing DataLoader 0: 100%|██████████████████████████████████████████████████████████| 157/157 [00:06<00:00, 26.03it/s]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      test_accuracy         0.8938000202178955
        test_loss           0.3923949599266052
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
============================================================
Final Test Accuracy: 0.8938
Target not reached, but close to 89.38%
============================================================
Model saved to: ./checkpoints\final_model.pth

Training completed!
```