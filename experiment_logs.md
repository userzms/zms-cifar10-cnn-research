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
