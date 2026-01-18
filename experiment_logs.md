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

## 实验4：微调后完整测试
- 时间：2026年1月15日
- 配置：batch_size=64, max_epochs=190
- 结果：测试准确率 90.94%
- 性能分析  
起始准确率：51.76%（Epoch 0）  
总提升：39.18个百分点
- 命令行输出：
```bash
(.venv) PS E:\python_exercises\zms_cifar10_cnn> python train.py --test

Running quick test...
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\torchvision\datasets\cifar.py:83: VisibleDeprecationWarning: dtype(): align should be passed as Python or NumPy boolean but got `align=0`. Did you mean to pass a tuple to create a subarray type? (Deprecated NumPy 2.4)
  entry = pickle.load(f, encoding="latin1")
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\torch\utils\data\dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
   Data loaded successfully!
   Batch shape: torch.Size([64, 3, 32, 32])
   Labels shape: torch.Size([64])
   Number of classes: 10
   Model forward pass successful!
   Output shape: torch.Size([64, 10])
(.venv) PS E:\python_exercises\zms_cifar10_cnn> python train.py       
============================================================
CIFAR10 CNN Training - Target: 93%+ Accuracy
============================================================
GPU Not Available, using CPU

Loading CIFAR10 dataset...
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\torchvision\datasets\cifar.py:83: VisibleDeprecationWarning: dtype(): align should be passed as Python or NumPy boolean but got `align=0`. Did you mean to pass a tuple to create a subarray type? (Deprecated NumPy 2.4)
  entry = pickle.load(f, encoding="latin1")
Initializing CNN model...
Model Parameters: Total=2,876,714, Trainable=2,876,714
GPU available: False, used: False
TPU available: False, using: 0 TPU cores

Starting training...
Max Epochs: 190
Target Accuracy: 93%+
============================================================
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:881: Checkpoint directory E:\python_exercises\zms_cifar10_cnn\checkpoints exists and is not empty.

   | Name      | Type               | Params | Mode  | FLOPs
------------------------------------------------------------------
0  | conv1     | Conv2d             | 2.6 K  | train | 0    
1  | bn1       | BatchNorm2d        | 192    | train | 0    
2  | conv2     | Conv2d             | 165 K  | train | 0    
3  | bn2       | BatchNorm2d        | 384    | train | 0    
4  | conv3     | Conv2d             | 663 K  | train | 0    
5  | bn3       | BatchNorm2d        | 768    | train | 0    
6  | conv4     | Conv2d             | 1.8 M  | train | 0    
7  | bn4       | BatchNorm2d        | 1.0 K  | train | 0    
8  | gap       | AdaptiveAvgPool2d  | 0      | train | 0    
9  | fc1       | Linear             | 196 K  | train | 0    
10 | dropout1  | Dropout            | 0      | train | 0    
11 | fc2       | Linear             | 73.9 K | train | 0    
12 | dropout2  | Dropout            | 0      | train | 0    
13 | fc3       | Linear             | 1.9 K  | train | 0    
14 | criterion | CrossEntropyLoss   | 0      | train | 0    
15 | train_acc | MulticlassAccuracy | 0      | train | 0    
16 | val_acc   | MulticlassAccuracy | 0      | train | 0    
17 | test_acc  | MulticlassAccuracy | 0      | train | 0    
------------------------------------------------------------------
2.9 M     Trainable params
0         Non-trainable params
2.9 M     Total params
11.507    Total estimated model params size (MB)
18        Modules in train mode
0         Modules in eval mode
0         Total Flops
Sanity Checking: |                                                                               | 0/? [00:00<?, ?it/s]E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=17` in the `DataLoader` to improve performance.
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\torch\utils\data\dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=17` in the `DataLoader` to improve performance.
Epoch 0: 100%|█| 782/782 [03:00<00:00,  4.32it/s, v_num=4, train_loss=1.300, train_acc=0.312, val_loss=1.410, val_accurMetric val_accuracy improved. New best score: 0.518                                                                     
Epoch 0, global step 782: 'val_accuracy' reached 0.51760 (best 0.51760), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=00-val_accuracy=0.5176.ckpt' as top 3
Epoch 1: 100%|█| 782/782 [03:15<00:00,  3.99it/s, v_num=4, train_loss=0.858, train_acc=0.750, val_loss=1.090, val_accurMetric val_accuracy improved by 0.097 >= min_delta = 0.0005. New best score: 0.614                                      
Epoch 1, global step 1564: 'val_accuracy' reached 0.61440 (best 0.61440), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=01-val_accuracy=0.6144.ckpt' as top 3
Epoch 2: 100%|█| 782/782 [03:20<00:00,  3.90it/s, v_num=4, train_loss=0.900, train_acc=0.625, val_loss=0.921, val_accurMetric val_accuracy improved by 0.060 >= min_delta = 0.0005. New best score: 0.674                                      
Epoch 2, global step 2346: 'val_accuracy' reached 0.67430 (best 0.67430), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=02-val_accuracy=0.6743.ckpt' as top 3
Epoch 3: 100%|█| 782/782 [03:19<00:00,  3.91it/s, v_num=4, train_loss=1.070, train_acc=0.500, val_loss=0.822, val_accurMetric val_accuracy improved by 0.033 >= min_delta = 0.0005. New best score: 0.708                                      
Epoch 3, global step 3128: 'val_accuracy' reached 0.70770 (best 0.70770), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=03-val_accuracy=0.7077.ckpt' as top 3
Epoch 4: 100%|█| 782/782 [03:11<00:00,  4.09it/s, v_num=4, train_loss=0.800, train_acc=0.750, val_loss=0.882, val_accurEpoch 4, global step 3910: 'val_accuracy' reached 0.69210 (best 0.70770), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=04-val_accuracy=0.6921.ckpt' as top 3
Epoch 5: 100%|█| 782/782 [03:11<00:00,  4.07it/s, v_num=4, train_loss=0.958, train_acc=0.750, val_loss=0.701, val_accurMetric val_accuracy improved by 0.048 >= min_delta = 0.0005. New best score: 0.756                                      
Epoch 5, global step 4692: 'val_accuracy' reached 0.75560 (best 0.75560), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=05-val_accuracy=0.7556.ckpt' as top 3
Epoch 6: 100%|█| 782/782 [03:21<00:00,  3.88it/s, v_num=4, train_loss=1.060, train_acc=0.750, val_loss=0.834, val_accurEpoch 6, global step 5474: 'val_accuracy' reached 0.70970 (best 0.75560), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=06-val_accuracy=0.7097.ckpt' as top 3
Epoch 7: 100%|█| 782/782 [03:21<00:00,  3.89it/s, v_num=4, train_loss=0.773, train_acc=0.750, val_loss=0.669, val_accurMetric val_accuracy improved by 0.012 >= min_delta = 0.0005. New best score: 0.767                                      
Epoch 7, global step 6256: 'val_accuracy' reached 0.76740 (best 0.76740), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=07-val_accuracy=0.7674.ckpt' as top 3
Epoch 8: 100%|█| 782/782 [03:21<00:00,  3.88it/s, v_num=4, train_loss=0.531, train_acc=0.750, val_loss=0.668, val_accurMetric val_accuracy improved by 0.008 >= min_delta = 0.0005. New best score: 0.775                                      
Epoch 8, global step 7038: 'val_accuracy' reached 0.77500 (best 0.77500), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=08-val_accuracy=0.7750.ckpt' as top 3
Epoch 9: 100%|█| 782/782 [03:12<00:00,  4.06it/s, v_num=4, train_loss=0.570, train_acc=0.938, val_loss=0.646, val_accurMetric val_accuracy improved by 0.005 >= min_delta = 0.0005. New best score: 0.780                                      
Epoch 9, global step 7820: 'val_accuracy' reached 0.78000 (best 0.78000), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=09-val_accuracy=0.7800.ckpt' as top 3
Epoch 10: 100%|█| 782/782 [02:52<00:00,  4.54it/s, v_num=4, train_loss=0.969, train_acc=0.625, val_loss=0.644, val_accuEpoch 10, global step 8602: 'val_accuracy' reached 0.77920 (best 0.78000), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=10-val_accuracy=0.7792.ckpt' as top 3
Epoch 11: 100%|█| 782/782 [02:52<00:00,  4.53it/s, v_num=4, train_loss=0.839, train_acc=0.688, val_loss=0.626, val_accuMetric val_accuracy improved by 0.008 >= min_delta = 0.0005. New best score: 0.788                                      
Epoch 11, global step 9384: 'val_accuracy' reached 0.78780 (best 0.78780), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=11-val_accuracy=0.7878.ckpt' as top 3
Epoch 12: 100%|█| 782/782 [02:52<00:00,  4.54it/s, v_num=4, train_loss=0.623, train_acc=0.812, val_loss=0.558, val_accuMetric val_accuracy improved by 0.027 >= min_delta = 0.0005. New best score: 0.815                                      
Epoch 12, global step 10166: 'val_accuracy' reached 0.81470 (best 0.81470), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=12-val_accuracy=0.8147.ckpt' as top 3
Epoch 13: 100%|█| 782/782 [02:52<00:00,  4.52it/s, v_num=4, train_loss=0.983, train_acc=0.625, val_loss=0.596, val_accuEpoch 13, global step 10948: 'val_accuracy' reached 0.80130 (best 0.81470), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=13-val_accuracy=0.8013.ckpt' as top 3
Epoch 14: 100%|█| 782/782 [02:52<00:00,  4.52it/s, v_num=4, train_loss=0.590, train_acc=0.812, val_loss=0.600, val_accuEpoch 14, global step 11730: 'val_accuracy' reached 0.79390 (best 0.81470), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=14-val_accuracy=0.7939.ckpt' as top 3
Epoch 15: 100%|█| 782/782 [02:59<00:00,  4.36it/s, v_num=4, train_loss=0.641, train_acc=0.750, val_loss=0.545, val_accuMetric val_accuracy improved by 0.003 >= min_delta = 0.0005. New best score: 0.818                                      
Epoch 15, global step 12512: 'val_accuracy' reached 0.81790 (best 0.81790), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=15-val_accuracy=0.8179.ckpt' as top 3
Epoch 16: 100%|█| 782/782 [02:59<00:00,  4.37it/s, v_num=4, train_loss=0.673, train_acc=0.750, val_loss=0.509, val_accuMetric val_accuracy improved by 0.010 >= min_delta = 0.0005. New best score: 0.828                                      
Epoch 16, global step 13294: 'val_accuracy' reached 0.82810 (best 0.82810), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=16-val_accuracy=0.8281.ckpt' as top 3
Epoch 17: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.415, train_acc=0.875, val_loss=0.465, val_accuMetric val_accuracy improved by 0.013 >= min_delta = 0.0005. New best score: 0.841                                      
Epoch 17, global step 14076: 'val_accuracy' reached 0.84150 (best 0.84150), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=17-val_accuracy=0.8415.ckpt' as top 3
Epoch 18: 100%|█| 782/782 [02:58<00:00,  4.38it/s, v_num=4, train_loss=0.522, train_acc=0.812, val_loss=0.486, val_accuEpoch 18, global step 14858: 'val_accuracy' reached 0.83880 (best 0.84150), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=18-val_accuracy=0.8388.ckpt' as top 3
Epoch 19: 100%|█| 782/782 [02:59<00:00,  4.36it/s, v_num=4, train_loss=0.649, train_acc=0.875, val_loss=0.538, val_accuEpoch 19, global step 15640: 'val_accuracy' was not in top 3                                                            
Epoch 20: 100%|█| 782/782 [03:02<00:00,  4.29it/s, v_num=4, train_loss=0.589, train_acc=0.812, val_loss=0.479, val_accuEpoch 20, global step 16422: 'val_accuracy' reached 0.84140 (best 0.84150), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=20-val_accuracy=0.8414.ckpt' as top 3
Epoch 21: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.107, train_acc=1.000, val_loss=0.445, val_accuMetric val_accuracy improved by 0.005 >= min_delta = 0.0005. New best score: 0.847                                      
Epoch 21, global step 17204: 'val_accuracy' reached 0.84670 (best 0.84670), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=21-val_accuracy=0.8467.ckpt' as top 3
Epoch 22: 100%|█| 782/782 [02:58<00:00,  4.39it/s, v_num=4, train_loss=0.765, train_acc=0.750, val_loss=0.457, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.848                                      
Epoch 22, global step 17986: 'val_accuracy' reached 0.84800 (best 0.84800), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=22-val_accuracy=0.8480.ckpt' as top 3
Epoch 23: 100%|█| 782/782 [03:00<00:00,  4.32it/s, v_num=4, train_loss=0.839, train_acc=0.625, val_loss=0.442, val_accuMetric val_accuracy improved by 0.006 >= min_delta = 0.0005. New best score: 0.854                                      
Epoch 23, global step 18768: 'val_accuracy' reached 0.85400 (best 0.85400), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=23-val_accuracy=0.8540.ckpt' as top 3
Epoch 24: 100%|█| 782/782 [03:00<00:00,  4.32it/s, v_num=4, train_loss=0.382, train_acc=0.875, val_loss=0.424, val_accuMetric val_accuracy improved by 0.003 >= min_delta = 0.0005. New best score: 0.857                                      
Epoch 24, global step 19550: 'val_accuracy' reached 0.85660 (best 0.85660), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=24-val_accuracy=0.8566.ckpt' as top 3
Epoch 25: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.433, train_acc=0.812, val_loss=0.437, val_accuEpoch 25, global step 20332: 'val_accuracy' reached 0.85350 (best 0.85660), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=25-val_accuracy=0.8535.ckpt' as top 3
Epoch 26: 100%|█| 782/782 [03:08<00:00,  4.14it/s, v_num=4, train_loss=0.760, train_acc=0.750, val_loss=0.442, val_accuEpoch 26, global step 21114: 'val_accuracy' was not in top 3                                                            
Epoch 27: 100%|█| 782/782 [03:01<00:00,  4.32it/s, v_num=4, train_loss=0.290, train_acc=0.938, val_loss=0.421, val_accuMetric val_accuracy improved by 0.004 >= min_delta = 0.0005. New best score: 0.860                                      
Epoch 27, global step 21896: 'val_accuracy' reached 0.86050 (best 0.86050), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=27-val_accuracy=0.8605.ckpt' as top 3
Epoch 28: 100%|█| 782/782 [23:23<00:00,  0.56it/s, v_num=4, train_loss=0.576, train_acc=0.812, val_loss=0.437, val_accuEpoch 28, global step 22678: 'val_accuracy' reached 0.85570 (best 0.86050), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=28-val_accuracy=0.8557.ckpt' as top 3
Epoch 29: 100%|█| 782/782 [02:57<00:00,  4.39it/s, v_num=4, train_loss=0.372, train_acc=0.875, val_loss=0.431, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.863                                      
Epoch 29, global step 23460: 'val_accuracy' reached 0.86280 (best 0.86280), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=29-val_accuracy=0.8628.ckpt' as top 3
Epoch 30: 100%|█| 782/782 [04:07<00:00,  3.16it/s, v_num=4, train_loss=0.299, train_acc=0.938, val_loss=0.395, val_accuMetric val_accuracy improved by 0.004 >= min_delta = 0.0005. New best score: 0.867                                      
Epoch 30, global step 24242: 'val_accuracy' reached 0.86690 (best 0.86690), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=30-val_accuracy=0.8669.ckpt' as top 3
Epoch 31: 100%|█| 782/782 [03:06<00:00,  4.20it/s, v_num=4, train_loss=0.529, train_acc=0.875, val_loss=0.396, val_accuMetric val_accuracy improved by 0.003 >= min_delta = 0.0005. New best score: 0.870                                      
Epoch 31, global step 25024: 'val_accuracy' reached 0.87030 (best 0.87030), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=31-val_accuracy=0.8703.ckpt' as top 3
Epoch 32: 100%|█| 782/782 [03:01<00:00,  4.30it/s, v_num=4, train_loss=0.550, train_acc=0.875, val_loss=0.398, val_accuEpoch 32, global step 25806: 'val_accuracy' reached 0.86720 (best 0.87030), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=32-val_accuracy=0.8672.ckpt' as top 3
Epoch 33: 100%|█| 782/782 [02:58<00:00,  4.37it/s, v_num=4, train_loss=0.615, train_acc=0.750, val_loss=0.415, val_accuEpoch 33, global step 26588: 'val_accuracy' was not in top 3                                                            
Epoch 34: 100%|█| 782/782 [02:59<00:00,  4.36it/s, v_num=4, train_loss=0.646, train_acc=0.875, val_loss=0.389, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.872                                      
Epoch 34, global step 27370: 'val_accuracy' reached 0.87190 (best 0.87190), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=34-val_accuracy=0.8719.ckpt' as top 3
Epoch 35: 100%|█| 782/782 [02:56<00:00,  4.42it/s, v_num=4, train_loss=0.617, train_acc=0.812, val_loss=0.412, val_accuEpoch 35, global step 28152: 'val_accuracy' was not in top 3                                                            
Epoch 36: 100%|█| 782/782 [02:58<00:00,  4.37it/s, v_num=4, train_loss=0.726, train_acc=0.688, val_loss=0.373, val_accuMetric val_accuracy improved by 0.008 >= min_delta = 0.0005. New best score: 0.880                                      
Epoch 36, global step 28934: 'val_accuracy' reached 0.87960 (best 0.87960), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=36-val_accuracy=0.8796.ckpt' as top 3
Epoch 37: 100%|█| 782/782 [02:58<00:00,  4.37it/s, v_num=4, train_loss=0.618, train_acc=0.812, val_loss=0.388, val_accuEpoch 37, global step 29716: 'val_accuracy' reached 0.87360 (best 0.87960), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=37-val_accuracy=0.8736.ckpt' as top 3
Epoch 38: 100%|█| 782/782 [02:59<00:00,  4.36it/s, v_num=4, train_loss=0.625, train_acc=0.875, val_loss=0.410, val_accuEpoch 38, global step 30498: 'val_accuracy' was not in top 3                                                            
Epoch 39: 100%|█| 782/782 [02:59<00:00,  4.36it/s, v_num=4, train_loss=0.482, train_acc=0.812, val_loss=0.381, val_accuEpoch 39, global step 31280: 'val_accuracy' reached 0.87830 (best 0.87960), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=39-val_accuracy=0.8783.ckpt' as top 3
Epoch 40: 100%|█| 782/782 [02:59<00:00,  4.36it/s, v_num=4, train_loss=0.329, train_acc=0.875, val_loss=0.375, val_accuEpoch 40, global step 32062: 'val_accuracy' reached 0.87860 (best 0.87960), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=40-val_accuracy=0.8786.ckpt' as top 3
Epoch 41: 100%|█| 782/782 [02:59<00:00,  4.37it/s, v_num=4, train_loss=0.171, train_acc=0.938, val_loss=0.378, val_accuEpoch 41, global step 32844: 'val_accuracy' was not in top 3                                                            
Epoch 42: 100%|█| 782/782 [02:57<00:00,  4.39it/s, v_num=4, train_loss=0.998, train_acc=0.625, val_loss=0.368, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.880                                      
Epoch 42, global step 33626: 'val_accuracy' reached 0.88020 (best 0.88020), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=42-val_accuracy=0.8802.ckpt' as top 3
Epoch 43: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.191, train_acc=0.938, val_loss=0.358, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.882                                      
Epoch 43, global step 34408: 'val_accuracy' reached 0.88190 (best 0.88190), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=43-val_accuracy=0.8819.ckpt' as top 3
Epoch 44: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.311, train_acc=0.875, val_loss=0.373, val_accuEpoch 44, global step 35190: 'val_accuracy' was not in top 3                                                            
Epoch 45: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.739, train_acc=0.812, val_loss=0.381, val_accuEpoch 45, global step 35972: 'val_accuracy' was not in top 3                                                            
Epoch 46: 100%|█| 782/782 [02:58<00:00,  4.37it/s, v_num=4, train_loss=0.542, train_acc=0.875, val_loss=0.353, val_accuMetric val_accuracy improved by 0.003 >= min_delta = 0.0005. New best score: 0.885                                      
Epoch 46, global step 36754: 'val_accuracy' reached 0.88490 (best 0.88490), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=46-val_accuracy=0.8849.ckpt' as top 3
Epoch 47: 100%|█| 782/782 [03:00<00:00,  4.34it/s, v_num=4, train_loss=0.630, train_acc=0.875, val_loss=0.365, val_accuEpoch 47, global step 37536: 'val_accuracy' reached 0.88450 (best 0.88490), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=47-val_accuracy=0.8845.ckpt' as top 3
Epoch 48: 100%|█| 782/782 [02:58<00:00,  4.37it/s, v_num=4, train_loss=0.338, train_acc=0.875, val_loss=0.361, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.887                                      
Epoch 48, global step 38318: 'val_accuracy' reached 0.88680 (best 0.88680), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=48-val_accuracy=0.8868.ckpt' as top 3
Epoch 49: 100%|█| 782/782 [02:58<00:00,  4.38it/s, v_num=4, train_loss=0.717, train_acc=0.750, val_loss=0.378, val_accuEpoch 49, global step 39100: 'val_accuracy' was not in top 3                                                            
Epoch 50: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.431, train_acc=0.812, val_loss=0.368, val_accuEpoch 50, global step 39882: 'val_accuracy' was not in top 3                                                            
Epoch 51: 100%|█| 782/782 [02:59<00:00,  4.37it/s, v_num=4, train_loss=0.351, train_acc=0.812, val_loss=0.361, val_accuEpoch 51, global step 40664: 'val_accuracy' reached 0.88610 (best 0.88680), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=51-val_accuracy=0.8861.ckpt' as top 3
Epoch 52: 100%|█| 782/782 [03:00<00:00,  4.34it/s, v_num=4, train_loss=0.482, train_acc=0.812, val_loss=0.366, val_accuEpoch 52, global step 41446: 'val_accuracy' was not in top 3                                                            
Epoch 53: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.292, train_acc=0.875, val_loss=0.362, val_accuEpoch 53, global step 42228: 'val_accuracy' was not in top 3                                                            
Epoch 54: 100%|█| 782/782 [02:58<00:00,  4.38it/s, v_num=4, train_loss=0.134, train_acc=0.938, val_loss=0.355, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.888                                      
Epoch 54, global step 43010: 'val_accuracy' reached 0.88780 (best 0.88780), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=54-val_accuracy=0.8878.ckpt' as top 3
Epoch 55: 100%|█| 782/782 [03:00<00:00,  4.34it/s, v_num=4, train_loss=0.456, train_acc=0.750, val_loss=0.375, val_accuEpoch 55, global step 43792: 'val_accuracy' was not in top 3                                                            
Epoch 56: 100%|█| 782/782 [02:58<00:00,  4.38it/s, v_num=4, train_loss=0.685, train_acc=0.750, val_loss=0.347, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.890                                      
Epoch 56, global step 44574: 'val_accuracy' reached 0.88960 (best 0.88960), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=56-val_accuracy=0.8896.ckpt' as top 3
Epoch 57: 100%|█| 782/782 [02:58<00:00,  4.37it/s, v_num=4, train_loss=0.134, train_acc=1.000, val_loss=0.371, val_accuEpoch 57, global step 45356: 'val_accuracy' was not in top 3                                                            
Epoch 58: 100%|█| 782/782 [05:17<00:00,  2.46it/s, v_num=4, train_loss=0.483, train_acc=0.875, val_loss=0.358, val_accuEpoch 58, global step 46138: 'val_accuracy' was not in top 3                                                            
Epoch 59: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.162, train_acc=0.938, val_loss=0.350, val_accuEpoch 59, global step 46920: 'val_accuracy' reached 0.88810 (best 0.88960), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=59-val_accuracy=0.8881.ckpt' as top 3
Epoch 60: 100%|█| 782/782 [03:01<00:00,  4.30it/s, v_num=4, train_loss=0.977, train_acc=0.688, val_loss=0.389, val_accuEpoch 60, global step 47702: 'val_accuracy' was not in top 3                                                            
Epoch 61: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.423, train_acc=0.750, val_loss=0.381, val_accuEpoch 61, global step 48484: 'val_accuracy' was not in top 3                                                            
Epoch 62: 100%|█| 782/782 [03:00<00:00,  4.34it/s, v_num=4, train_loss=0.0424, train_acc=1.000, val_loss=0.366, val_accEpoch 62, global step 49266: 'val_accuracy' reached 0.88790 (best 0.88960), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=62-val_accuracy=0.8879.ckpt' as top 3
Epoch 63: 100%|█| 782/782 [03:01<00:00,  4.32it/s, v_num=4, train_loss=0.365, train_acc=0.875, val_loss=0.347, val_accuMetric val_accuracy improved by 0.005 >= min_delta = 0.0005. New best score: 0.894                                      
Epoch 63, global step 50048: 'val_accuracy' reached 0.89410 (best 0.89410), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=63-val_accuracy=0.8941.ckpt' as top 3
Epoch 64: 100%|█| 782/782 [03:01<00:00,  4.30it/s, v_num=4, train_loss=0.212, train_acc=0.938, val_loss=0.362, val_accuEpoch 64, global step 50830: 'val_accuracy' reached 0.88970 (best 0.89410), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=64-val_accuracy=0.8897.ckpt' as top 3
Epoch 65: 100%|█| 782/782 [02:56<00:00,  4.43it/s, v_num=4, train_loss=0.468, train_acc=0.812, val_loss=0.375, val_accuEpoch 65, global step 51612: 'val_accuracy' was not in top 3                                                            
Epoch 66: 100%|█| 782/782 [02:53<00:00,  4.52it/s, v_num=4, train_loss=0.435, train_acc=0.875, val_loss=0.354, val_accuEpoch 66, global step 52394: 'val_accuracy' was not in top 3                                                            
Epoch 67: 100%|█| 782/782 [10:05<00:00,  1.29it/s, v_num=4, train_loss=0.187, train_acc=0.938, val_loss=0.349, val_accuEpoch 67, global step 53176: 'val_accuracy' reached 0.89200 (best 0.89410), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=67-val_accuracy=0.8920.ckpt' as top 3
Epoch 68: 100%|█| 782/782 [02:56<00:00,  4.44it/s, v_num=4, train_loss=0.440, train_acc=0.812, val_loss=0.355, val_accuEpoch 68, global step 53958: 'val_accuracy' reached 0.89090 (best 0.89410), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=68-val_accuracy=0.8909.ckpt' as top 3
Epoch 69: 100%|█| 782/782 [02:55<00:00,  4.45it/s, v_num=4, train_loss=0.106, train_acc=1.000, val_loss=0.374, val_accuEpoch 69, global step 54740: 'val_accuracy' was not in top 3                                                            
Epoch 70: 100%|█| 782/782 [02:54<00:00,  4.49it/s, v_num=4, train_loss=0.219, train_acc=0.938, val_loss=0.371, val_accuEpoch 70, global step 55522: 'val_accuracy' was not in top 3                                                            
Epoch 71: 100%|█| 782/782 [02:53<00:00,  4.51it/s, v_num=4, train_loss=0.0953, train_acc=0.938, val_loss=0.366, val_accEpoch 71, global step 56304: 'val_accuracy' reached 0.89100 (best 0.89410), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=71-val_accuracy=0.8910.ckpt' as top 3
Epoch 72: 100%|█| 782/782 [02:52<00:00,  4.53it/s, v_num=4, train_loss=0.317, train_acc=0.875, val_loss=0.359, val_accuEpoch 72, global step 57086: 'val_accuracy' reached 0.89290 (best 0.89410), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=72-val_accuracy=0.8929.ckpt' as top 3
Epoch 73: 100%|█| 782/782 [02:53<00:00,  4.52it/s, v_num=4, train_loss=0.474, train_acc=0.812, val_loss=0.358, val_accuEpoch 73, global step 57868: 'val_accuracy' reached 0.89370 (best 0.89410), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=73-val_accuracy=0.8937.ckpt' as top 3
Epoch 74: 100%|█| 782/782 [02:51<00:00,  4.55it/s, v_num=4, train_loss=0.472, train_acc=0.812, val_loss=0.376, val_accuEpoch 74, global step 58650: 'val_accuracy' was not in top 3                                                            
Epoch 75: 100%|█| 782/782 [03:02<00:00,  4.30it/s, v_num=4, train_loss=0.0486, train_acc=1.000, val_loss=0.360, val_accEpoch 75, global step 59432: 'val_accuracy' was not in top 3                                                            
Epoch 76: 100%|█| 782/782 [02:53<00:00,  4.51it/s, v_num=4, train_loss=0.173, train_acc=0.938, val_loss=0.379, val_accuEpoch 76, global step 60214: 'val_accuracy' was not in top 3                                                            
Epoch 77: 100%|█| 782/782 [02:53<00:00,  4.50it/s, v_num=4, train_loss=0.610, train_acc=0.750, val_loss=0.359, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.896                                      
Epoch 77, global step 60996: 'val_accuracy' reached 0.89620 (best 0.89620), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=77-val_accuracy=0.8962.ckpt' as top 3
Epoch 78: 100%|█| 782/782 [03:46<00:00,  3.46it/s, v_num=4, train_loss=0.183, train_acc=1.000, val_loss=0.363, val_accuEpoch 78, global step 61778: 'val_accuracy' reached 0.89630 (best 0.89630), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=78-val_accuracy=0.8963.ckpt' as top 3
Epoch 79: 100%|█| 782/782 [03:15<00:00,  3.99it/s, v_num=4, train_loss=0.136, train_acc=0.875, val_loss=0.376, val_accuEpoch 79, global step 62560: 'val_accuracy' was not in top 3                                                            
Epoch 80: 100%|█| 782/782 [02:57<00:00,  4.41it/s, v_num=4, train_loss=0.356, train_acc=0.875, val_loss=0.365, val_accuEpoch 80, global step 63342: 'val_accuracy' reached 0.89640 (best 0.89640), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=80-val_accuracy=0.8964.ckpt' as top 3
Epoch 81: 100%|█| 782/782 [03:14<00:00,  4.01it/s, v_num=4, train_loss=0.293, train_acc=0.938, val_loss=0.360, val_accuEpoch 81, global step 64124: 'val_accuracy' was not in top 3                                                            
Epoch 82: 100%|█| 782/782 [03:20<00:00,  3.90it/s, v_num=4, train_loss=0.428, train_acc=0.812, val_loss=0.364, val_accuEpoch 82, global step 64906: 'val_accuracy' was not in top 3                                                            
Epoch 83: 100%|█| 782/782 [03:20<00:00,  3.90it/s, v_num=4, train_loss=0.226, train_acc=0.875, val_loss=0.368, val_accuEpoch 83, global step 65688: 'val_accuracy' was not in top 3                                                            
Epoch 84: 100%|█| 782/782 [03:22<00:00,  3.86it/s, v_num=4, train_loss=0.302, train_acc=0.938, val_loss=0.377, val_accuEpoch 84, global step 66470: 'val_accuracy' was not in top 3                                                            
Epoch 85: 100%|█| 782/782 [03:12<00:00,  4.07it/s, v_num=4, train_loss=0.599, train_acc=0.812, val_loss=0.361, val_accuMetric val_accuracy improved by 0.003 >= min_delta = 0.0005. New best score: 0.899                                      
Epoch 85, global step 67252: 'val_accuracy' reached 0.89930 (best 0.89930), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=85-val_accuracy=0.8993.ckpt' as top 3
Epoch 86: 100%|█| 782/782 [03:12<00:00,  4.06it/s, v_num=4, train_loss=0.312, train_acc=0.875, val_loss=0.369, val_accuEpoch 86, global step 68034: 'val_accuracy' was not in top 3                                                            
Epoch 87: 100%|█| 782/782 [04:42<00:00,  2.77it/s, v_num=4, train_loss=0.229, train_acc=0.875, val_loss=0.367, val_accuEpoch 87, global step 68816: 'val_accuracy' reached 0.89910 (best 0.89930), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=87-val_accuracy=0.8991.ckpt' as top 3
Epoch 88: 100%|█| 782/782 [03:13<00:00,  4.05it/s, v_num=4, train_loss=0.120, train_acc=1.000, val_loss=0.353, val_accuEpoch 88, global step 69598: 'val_accuracy' reached 0.89660 (best 0.89930), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=88-val_accuracy=0.8966.ckpt' as top 3
Epoch 89: 100%|█| 782/782 [03:19<00:00,  3.92it/s, v_num=4, train_loss=0.248, train_acc=0.938, val_loss=0.352, val_accuEpoch 89, global step 70380: 'val_accuracy' reached 0.89830 (best 0.89930), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=89-val_accuracy=0.8983.ckpt' as top 3
Epoch 90: 100%|█| 782/782 [02:54<00:00,  4.48it/s, v_num=4, train_loss=0.146, train_acc=0.875, val_loss=0.352, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.900                                      
Epoch 90, global step 71162: 'val_accuracy' reached 0.90000 (best 0.90000), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=90-val_accuracy=0.9000.ckpt' as top 3
Epoch 91: 100%|█| 782/782 [02:54<00:00,  4.47it/s, v_num=4, train_loss=0.295, train_acc=0.875, val_loss=0.372, val_accuEpoch 91, global step 71944: 'val_accuracy' was not in top 3                                                            
Epoch 92: 100%|█| 782/782 [02:54<00:00,  4.49it/s, v_num=4, train_loss=0.733, train_acc=0.812, val_loss=0.383, val_accuEpoch 92, global step 72726: 'val_accuracy' was not in top 3                                                            
Epoch 93: 100%|█| 782/782 [02:53<00:00,  4.51it/s, v_num=4, train_loss=0.624, train_acc=0.688, val_loss=0.399, val_accuEpoch 93, global step 73508: 'val_accuracy' was not in top 3                                                            
Epoch 94: 100%|█| 782/782 [03:16<00:00,  3.98it/s, v_num=4, train_loss=0.281, train_acc=0.812, val_loss=0.361, val_accuEpoch 94, global step 74290: 'val_accuracy' was not in top 3                                                            
Epoch 95: 100%|█| 782/782 [03:10<00:00,  4.10it/s, v_num=4, train_loss=0.219, train_acc=0.875, val_loss=0.368, val_accuEpoch 95, global step 75072: 'val_accuracy' reached 0.90010 (best 0.90010), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=95-val_accuracy=0.9001.ckpt' as top 3
Epoch 96: 100%|█| 782/782 [03:20<00:00,  3.89it/s, v_num=4, train_loss=0.334, train_acc=0.812, val_loss=0.358, val_accuEpoch 96, global step 75854: 'val_accuracy' reached 0.89940 (best 0.90010), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=96-val_accuracy=0.8994.ckpt' as top 3
Epoch 97: 100%|█| 782/782 [03:05<00:00,  4.21it/s, v_num=4, train_loss=0.0274, train_acc=1.000, val_loss=0.357, val_accMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.901                                      
Epoch 97, global step 76636: 'val_accuracy' reached 0.90060 (best 0.90060), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=97-val_accuracy=0.9006.ckpt' as top 3
Epoch 98: 100%|█| 782/782 [03:03<00:00,  4.25it/s, v_num=4, train_loss=0.400, train_acc=0.938, val_loss=0.359, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.902                                      
Epoch 98, global step 77418: 'val_accuracy' reached 0.90160 (best 0.90160), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=98-val_accuracy=0.9016.ckpt' as top 3
Epoch 99: 100%|█| 782/782 [03:36<00:00,  3.62it/s, v_num=4, train_loss=0.302, train_acc=0.875, val_loss=0.366, val_accuEpoch 99, global step 78200: 'val_accuracy' was not in top 3                                                            
Epoch 100: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.352, train_acc=0.875, val_loss=0.359, val_accEpoch 100, global step 78982: 'val_accuracy' reached 0.90130 (best 0.90160), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=100-val_accuracy=0.9013.ckpt' as top 3
Epoch 101: 100%|█| 782/782 [02:58<00:00,  4.37it/s, v_num=4, train_loss=0.354, train_acc=0.812, val_loss=0.390, val_accEpoch 101, global step 79764: 'val_accuracy' was not in top 3                                                           
Epoch 102: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.551, train_acc=0.875, val_loss=0.384, val_accEpoch 102, global step 80546: 'val_accuracy' was not in top 3                                                           
Epoch 103: 100%|█| 782/782 [02:58<00:00,  4.38it/s, v_num=4, train_loss=0.485, train_acc=0.875, val_loss=0.385, val_accEpoch 103, global step 81328: 'val_accuracy' was not in top 3                                                           
Epoch 104: 100%|█| 782/782 [02:58<00:00,  4.37it/s, v_num=4, train_loss=0.146, train_acc=0.938, val_loss=0.385, val_accEpoch 104, global step 82110: 'val_accuracy' was not in top 3                                                           
Epoch 105: 100%|█| 782/782 [02:59<00:00,  4.37it/s, v_num=4, train_loss=0.0565, train_acc=1.000, val_loss=0.370, val_acEpoch 105, global step 82892: 'val_accuracy' was not in top 3                                                           
Epoch 106: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.738, train_acc=0.875, val_loss=0.368, val_accEpoch 106, global step 83674: 'val_accuracy' reached 0.90190 (best 0.90190), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=106-val_accuracy=0.9019.ckpt' as top 3
Epoch 107: 100%|█| 782/782 [02:58<00:00,  4.38it/s, v_num=4, train_loss=0.0939, train_acc=1.000, val_loss=0.373, val_acMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.904                                      
Epoch 107, global step 84456: 'val_accuracy' reached 0.90380 (best 0.90380), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=107-val_accuracy=0.9038.ckpt' as top 3
Epoch 108: 100%|█| 782/782 [02:59<00:00,  4.36it/s, v_num=4, train_loss=0.0303, train_acc=1.000, val_loss=0.399, val_acEpoch 108, global step 85238: 'val_accuracy' was not in top 3                                                           
Epoch 109: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.024, train_acc=1.000, val_loss=0.387, val_accEpoch 109, global step 86020: 'val_accuracy' was not in top 3                                                           
Epoch 110: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.115, train_acc=0.938, val_loss=0.388, val_accEpoch 110, global step 86802: 'val_accuracy' was not in top 3                                                           
Epoch 111: 100%|█| 782/782 [02:59<00:00,  4.36it/s, v_num=4, train_loss=0.0721, train_acc=0.938, val_loss=0.394, val_acEpoch 111, global step 87584: 'val_accuracy' was not in top 3                                                           
Epoch 112: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.135, train_acc=0.938, val_loss=0.386, val_accEpoch 112, global step 88366: 'val_accuracy' was not in top 3                                                           
Epoch 113: 100%|█| 782/782 [02:59<00:00,  4.34it/s, v_num=4, train_loss=0.247, train_acc=0.875, val_loss=0.390, val_accEpoch 113, global step 89148: 'val_accuracy' was not in top 3                                                           
Epoch 114: 100%|█| 782/782 [02:59<00:00,  4.36it/s, v_num=4, train_loss=0.326, train_acc=0.750, val_loss=0.374, val_accEpoch 114, global step 89930: 'val_accuracy' was not in top 3                                                           
Epoch 115: 100%|█| 782/782 [02:59<00:00,  4.36it/s, v_num=4, train_loss=0.145, train_acc=1.000, val_loss=0.381, val_accEpoch 115, global step 90712: 'val_accuracy' was not in top 3                                                           
Epoch 116: 100%|█| 782/782 [02:59<00:00,  4.36it/s, v_num=4, train_loss=0.0725, train_acc=1.000, val_loss=0.377, val_acEpoch 116, global step 91494: 'val_accuracy' reached 0.90260 (best 0.90380), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=116-val_accuracy=0.9026.ckpt' as top 3
Epoch 117: 100%|█| 782/782 [02:59<00:00,  4.36it/s, v_num=4, train_loss=0.0353, train_acc=1.000, val_loss=0.387, val_acEpoch 117, global step 92276: 'val_accuracy' was not in top 3                                                           
Epoch 118: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.141, train_acc=0.938, val_loss=0.386, val_accEpoch 118, global step 93058: 'val_accuracy' was not in top 3                                                           
Epoch 119: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.110, train_acc=0.938, val_loss=0.407, val_accEpoch 119, global step 93840: 'val_accuracy' reached 0.90270 (best 0.90380), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=119-val_accuracy=0.9027.ckpt' as top 3
Epoch 120: 100%|█| 782/782 [03:00<00:00,  4.34it/s, v_num=4, train_loss=0.0836, train_acc=0.938, val_loss=0.394, val_acEpoch 120, global step 94622: 'val_accuracy' reached 0.90350 (best 0.90380), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=120-val_accuracy=0.9035.ckpt' as top 3
Epoch 121: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.0355, train_acc=1.000, val_loss=0.410, val_acEpoch 121, global step 95404: 'val_accuracy' was not in top 3                                                           
Epoch 122: 100%|█| 782/782 [02:59<00:00,  4.36it/s, v_num=4, train_loss=0.630, train_acc=0.688, val_loss=0.394, val_accEpoch 122, global step 96186: 'val_accuracy' was not in top 3                                                           
Epoch 123: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.0365, train_acc=1.000, val_loss=0.402, val_acEpoch 123, global step 96968: 'val_accuracy' was not in top 3                                                           
Epoch 124: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.306, train_acc=0.875, val_loss=0.404, val_accEpoch 124, global step 97750: 'val_accuracy' was not in top 3                                                           
Epoch 125: 100%|█| 782/782 [03:00<00:00,  4.34it/s, v_num=4, train_loss=0.0898, train_acc=0.938, val_loss=0.387, val_acEpoch 125, global step 98532: 'val_accuracy' reached 0.90350 (best 0.90380), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=125-val_accuracy=0.9035.ckpt' as top 3
Epoch 126: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.0959, train_acc=0.938, val_loss=0.391, val_acEpoch 126, global step 99314: 'val_accuracy' was not in top 3                                                           
Epoch 127: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.220, train_acc=0.875, val_loss=0.395, val_accEpoch 127, global step 100096: 'val_accuracy' was not in top 3                                                          
Epoch 128: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.137, train_acc=0.875, val_loss=0.392, val_accEpoch 128, global step 100878: 'val_accuracy' was not in top 3                                                          
Epoch 129: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.151, train_acc=0.938, val_loss=0.413, val_accEpoch 129, global step 101660: 'val_accuracy' was not in top 3                                                          
Epoch 130: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.0784, train_acc=1.000, val_loss=0.404, val_acEpoch 130, global step 102442: 'val_accuracy' was not in top 3                                                          
Epoch 131: 100%|█| 782/782 [03:00<00:00,  4.32it/s, v_num=4, train_loss=0.0755, train_acc=0.938, val_loss=0.407, val_acEpoch 131, global step 103224: 'val_accuracy' was not in top 3                                                          
Epoch 132: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.0078, train_acc=1.000, val_loss=0.414, val_acMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.904                                      
Epoch 132, global step 104006: 'val_accuracy' reached 0.90440 (best 0.90440), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=132-val_accuracy=0.9044.ckpt' as top 3
Epoch 133: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.734, train_acc=0.812, val_loss=0.399, val_accMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.906                                      
Epoch 133, global step 104788: 'val_accuracy' reached 0.90570 (best 0.90570), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=133-val_accuracy=0.9057.ckpt' as top 3
Epoch 134: 100%|█| 782/782 [03:00<00:00,  4.34it/s, v_num=4, train_loss=0.0463, train_acc=1.000, val_loss=0.411, val_acMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.906                                      
Epoch 134, global step 105570: 'val_accuracy' reached 0.90630 (best 0.90630), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=134-val_accuracy=0.9063.ckpt' as top 3
Epoch 135: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.627, train_acc=0.938, val_loss=0.409, val_accEpoch 135, global step 106352: 'val_accuracy' reached 0.90460 (best 0.90630), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=135-val_accuracy=0.9046.ckpt' as top 3
Epoch 136: 100%|█| 782/782 [03:01<00:00,  4.30it/s, v_num=4, train_loss=0.105, train_acc=0.938, val_loss=0.404, val_accEpoch 136, global step 107134: 'val_accuracy' was not in top 3                                                          
Epoch 137: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.0591, train_acc=1.000, val_loss=0.396, val_acEpoch 137, global step 107916: 'val_accuracy' was not in top 3                                                          
Epoch 138: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.201, train_acc=0.938, val_loss=0.407, val_accEpoch 138, global step 108698: 'val_accuracy' reached 0.90600 (best 0.90630), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=138-val_accuracy=0.9060.ckpt' as top 3
Epoch 139: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.932, train_acc=0.812, val_loss=0.397, val_accEpoch 139, global step 109480: 'val_accuracy' was not in top 3                                                          
Epoch 140: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.137, train_acc=0.938, val_loss=0.419, val_accEpoch 140, global step 110262: 'val_accuracy' was not in top 3                                                          
Epoch 141: 100%|█| 782/782 [03:00<00:00,  4.32it/s, v_num=4, train_loss=0.0381, train_acc=1.000, val_loss=0.413, val_acEpoch 141, global step 111044: 'val_accuracy' reached 0.90620 (best 0.90630), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=141-val_accuracy=0.9062.ckpt' as top 3
Epoch 142: 100%|█| 782/782 [03:01<00:00,  4.32it/s, v_num=4, train_loss=0.0245, train_acc=1.000, val_loss=0.407, val_acEpoch 142, global step 111826: 'val_accuracy' was not in top 3                                                          
Epoch 143: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.399, train_acc=0.875, val_loss=0.404, val_accEpoch 143, global step 112608: 'val_accuracy' was not in top 3                                                          
Epoch 144: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.572, train_acc=0.750, val_loss=0.401, val_accEpoch 144, global step 113390: 'val_accuracy' was not in top 3                                                          
Epoch 145: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.0706, train_acc=1.000, val_loss=0.409, val_acEpoch 145, global step 114172: 'val_accuracy' was not in top 3                                                          
Epoch 146: 100%|█| 782/782 [02:57<00:00,  4.41it/s, v_num=4, train_loss=0.115, train_acc=1.000, val_loss=0.421, val_accEpoch 146, global step 114954: 'val_accuracy' was not in top 3                                                          
Epoch 147: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.152, train_acc=0.938, val_loss=0.416, val_accEpoch 147, global step 115736: 'val_accuracy' was not in top 3                                                          
Epoch 148: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.0704, train_acc=1.000, val_loss=0.408, val_acEpoch 148, global step 116518: 'val_accuracy' reached 0.90620 (best 0.90630), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=148-val_accuracy=0.9062.ckpt' as top 3
Epoch 149: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.112, train_acc=0.938, val_loss=0.417, val_accEpoch 149, global step 117300: 'val_accuracy' was not in top 3                                                          
Epoch 150: 100%|█| 782/782 [02:58<00:00,  4.37it/s, v_num=4, train_loss=0.237, train_acc=0.938, val_loss=0.412, val_accMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.908                                      
Epoch 150, global step 118082: 'val_accuracy' reached 0.90790 (best 0.90790), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=150-val_accuracy=0.9079.ckpt' as top 3
Epoch 151: 100%|█| 782/782 [03:02<00:00,  4.28it/s, v_num=4, train_loss=0.00896, train_acc=1.000, val_loss=0.409, val_aEpoch 151, global step 118864: 'val_accuracy' was not in top 3                                                          
Epoch 152: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.0685, train_acc=0.938, val_loss=0.417, val_acEpoch 152, global step 119646: 'val_accuracy' reached 0.90730 (best 0.90790), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=152-val_accuracy=0.9073.ckpt' as top 3
Epoch 153: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.150, train_acc=0.938, val_loss=0.412, val_accEpoch 153, global step 120428: 'val_accuracy' reached 0.90760 (best 0.90790), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=153-val_accuracy=0.9076.ckpt' as top 3
Epoch 154: 100%|█| 782/782 [03:00<00:00,  4.34it/s, v_num=4, train_loss=0.0107, train_acc=1.000, val_loss=0.409, val_acMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.909                                      
Epoch 154, global step 121210: 'val_accuracy' reached 0.90850 (best 0.90850), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=154-val_accuracy=0.9085.ckpt' as top 3
Epoch 155: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.254, train_acc=0.875, val_loss=0.413, val_accEpoch 155, global step 121992: 'val_accuracy' reached 0.90820 (best 0.90850), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=155-val_accuracy=0.9082.ckpt' as top 3
Epoch 156: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.0491, train_acc=1.000, val_loss=0.413, val_acEpoch 156, global step 122774: 'val_accuracy' was not in top 3                                                          
Epoch 157: 100%|█| 782/782 [03:00<00:00,  4.34it/s, v_num=4, train_loss=0.120, train_acc=0.938, val_loss=0.418, val_accEpoch 157, global step 123556: 'val_accuracy' was not in top 3                                                          
Epoch 158: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.0186, train_acc=1.000, val_loss=0.413, val_acEpoch 158, global step 124338: 'val_accuracy' was not in top 3                                                          
Epoch 159: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.0964, train_acc=1.000, val_loss=0.408, val_acEpoch 159, global step 125120: 'val_accuracy' was not in top 3                                                          
Epoch 160: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.426, train_acc=0.875, val_loss=0.417, val_accEpoch 160, global step 125902: 'val_accuracy' was not in top 3                                                          
Epoch 161: 100%|█| 782/782 [03:01<00:00,  4.32it/s, v_num=4, train_loss=0.0135, train_acc=1.000, val_loss=0.424, val_acEpoch 161, global step 126684: 'val_accuracy' reached 0.90850 (best 0.90850), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=161-val_accuracy=0.9085.ckpt' as top 3
Epoch 162: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.0487, train_acc=1.000, val_loss=0.421, val_acEpoch 162, global step 127466: 'val_accuracy' was not in top 3                                                          
Epoch 163: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.125, train_acc=0.938, val_loss=0.421, val_accEpoch 163, global step 128248: 'val_accuracy' was not in top 3                                                          
Epoch 164: 100%|█| 782/782 [03:00<00:00,  4.32it/s, v_num=4, train_loss=0.0115, train_acc=1.000, val_loss=0.420, val_acEpoch 164, global step 129030: 'val_accuracy' was not in top 3                                                          
Epoch 165: 100%|█| 782/782 [03:01<00:00,  4.32it/s, v_num=4, train_loss=0.333, train_acc=0.875, val_loss=0.429, val_accEpoch 165, global step 129812: 'val_accuracy' reached 0.90830 (best 0.90850), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=165-val_accuracy=0.9083.ckpt' as top 3
Epoch 166: 100%|█| 782/782 [03:01<00:00,  4.32it/s, v_num=4, train_loss=0.149, train_acc=0.938, val_loss=0.429, val_accEpoch 166, global step 130594: 'val_accuracy' was not in top 3                                                          
Epoch 167: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.299, train_acc=0.938, val_loss=0.430, val_accEpoch 167, global step 131376: 'val_accuracy' reached 0.90840 (best 0.90850), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=167-val_accuracy=0.9084.ckpt' as top 3
Epoch 168: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.0607, train_acc=1.000, val_loss=0.431, val_acMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.909                                      
Epoch 168, global step 132158: 'val_accuracy' reached 0.90930 (best 0.90930), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=168-val_accuracy=0.9093.ckpt' as top 3
Epoch 169: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.0627, train_acc=1.000, val_loss=0.431, val_acEpoch 169, global step 132940: 'val_accuracy' was not in top 3                                                          
Epoch 170: 100%|█| 782/782 [03:01<00:00,  4.32it/s, v_num=4, train_loss=0.230, train_acc=0.938, val_loss=0.423, val_accEpoch 170, global step 133722: 'val_accuracy' reached 0.90860 (best 0.90930), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=170-val_accuracy=0.9086.ckpt' as top 3
Epoch 171: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.0586, train_acc=1.000, val_loss=0.426, val_acEpoch 171, global step 134504: 'val_accuracy' was not in top 3                                                          
Epoch 172: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.413, train_acc=0.812, val_loss=0.423, val_accEpoch 172, global step 135286: 'val_accuracy' was not in top 3                                                          
Epoch 173: 100%|█| 782/782 [03:00<00:00,  4.32it/s, v_num=4, train_loss=0.0885, train_acc=1.000, val_loss=0.425, val_acEpoch 173, global step 136068: 'val_accuracy' reached 0.90930 (best 0.90930), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=173-val_accuracy=0.9093.ckpt' as top 3
Epoch 174: 100%|█| 782/782 [03:00<00:00,  4.32it/s, v_num=4, train_loss=0.380, train_acc=0.875, val_loss=0.424, val_accEpoch 174, global step 136850: 'val_accuracy' reached 0.90940 (best 0.90940), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=174-val_accuracy=0.9094.ckpt' as top 3
Epoch 175: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.0645, train_acc=1.000, val_loss=0.422, val_acEpoch 175, global step 137632: 'val_accuracy' was not in top 3                                                          
Epoch 176: 100%|█| 782/782 [03:01<00:00,  4.32it/s, v_num=4, train_loss=0.396, train_acc=0.938, val_loss=0.423, val_accEpoch 176, global step 138414: 'val_accuracy' was not in top 3                                                          
Epoch 177: 100%|█| 782/782 [03:00<00:00,  4.32it/s, v_num=4, train_loss=0.0238, train_acc=1.000, val_loss=0.425, val_acEpoch 177, global step 139196: 'val_accuracy' was not in top 3                                                          
Epoch 178: 100%|█| 782/782 [03:00<00:00,  4.32it/s, v_num=4, train_loss=0.115, train_acc=0.938, val_loss=0.423, val_accEpoch 178, global step 139978: 'val_accuracy' was not in top 3                                                          
Epoch 179: 100%|█| 782/782 [02:59<00:00,  4.35it/s, v_num=4, train_loss=0.102, train_acc=1.000, val_loss=0.423, val_accEpoch 179, global step 140760: 'val_accuracy' was not in top 3                                                          
Epoch 180: 100%|█| 782/782 [03:00<00:00,  4.32it/s, v_num=4, train_loss=0.0145, train_acc=1.000, val_loss=0.421, val_acEpoch 180, global step 141542: 'val_accuracy' was not in top 3                                                          
Epoch 181: 100%|█| 782/782 [03:01<00:00,  4.32it/s, v_num=4, train_loss=0.192, train_acc=0.938, val_loss=0.419, val_accEpoch 181, global step 142324: 'val_accuracy' was not in top 3                                                          
Epoch 182: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.0504, train_acc=1.000, val_loss=0.424, val_acEpoch 182, global step 143106: 'val_accuracy' was not in top 3                                                          
Epoch 183: 100%|█| 782/782 [03:02<00:00,  4.29it/s, v_num=4, train_loss=0.0207, train_acc=1.000, val_loss=0.429, val_acEpoch 183, global step 143888: 'val_accuracy' was not in top 3                                                          
Epoch 184: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.0333, train_acc=1.000, val_loss=0.426, val_acEpoch 184, global step 144670: 'val_accuracy' was not in top 3                                                          
Epoch 185: 100%|█| 782/782 [03:00<00:00,  4.32it/s, v_num=4, train_loss=0.469, train_acc=0.875, val_loss=0.428, val_accEpoch 185, global step 145452: 'val_accuracy' was not in top 3                                                          
Epoch 186: 100%|█| 782/782 [03:01<00:00,  4.32it/s, v_num=4, train_loss=0.0612, train_acc=1.000, val_loss=0.420, val_acEpoch 186, global step 146234: 'val_accuracy' was not in top 3                                                          
Epoch 187: 100%|█| 782/782 [03:00<00:00,  4.33it/s, v_num=4, train_loss=0.234, train_acc=0.938, val_loss=0.422, val_accEpoch 187, global step 147016: 'val_accuracy' was not in top 3                                                          
Epoch 188: 100%|█| 782/782 [03:01<00:00,  4.31it/s, v_num=4, train_loss=0.311, train_acc=0.875, val_loss=0.419, val_accEpoch 188, global step 147798: 'val_accuracy' was not in top 3                                                          
Epoch 189: 100%|█| 782/782 [03:00<00:00,  4.32it/s, v_num=4, train_loss=0.0166, train_acc=1.000, val_loss=0.417, val_acEpoch 189, global step 148580: 'val_accuracy' was not in top 3                                                          
`Trainer.fit` stopped: `max_epochs=190` reached.
Epoch 189: 100%|█| 782/782 [03:00<00:00,  4.32it/s, v_num=4, train_loss=0.0166, train_acc=1.000, val_loss=0.417, val_ac

Testing best model...
Loading best model: E:\python_exercises\zms_cifar10_cnn\checkpoints\cifar10-cnn-epoch=174-val_accuracy=0.9094.ckpt
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:434: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=17` in the `DataLoader` to improve performance.
Testing DataLoader 0: 100%|██████████████████████████████████████████████████████████| 157/157 [00:09<00:00, 16.55it/s]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      test_accuracy         0.9093999862670898
        test_loss           0.42416152358055115
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
============================================================
Final Test Accuracy: 0.9094
Target not reached, but close to 90.94%
============================================================
Model saved to: ./checkpoints\final_model.pth

Training completed!
```

- 问题分析：
    - 训练时间过长，但提升有限：190个epoch后，验证准确率仅达到90.94%，后期提升非常缓慢
    - 学习率策略不够激进：余弦退火学习率在后期学习率过低，无法有效突破瓶颈

## 实验5：优化后的完整测试
- 时间：2026年1月16日
- 配置：batch_size=64, max_epochs=100
- 结果：测试准确率 91.20%
- 性能分析  
起始准确率：50.16%（Epoch 0）  
总提升：41.04个百分点
- 做出的改进：  
1.在CNN模型中加入CBAM（Convolutional Block Attention Module）注意力机制，提升模型性能而不大幅增加计算成本。CBAM通过空间注意力和通道注意力两种机制，让模型学会关注更重要的特征区域和通道，对于细粒度分类任务（如CIFAR10）有效。  
2.优化学习率调度器，使用OneCycleLR策略，在训练初期快速提高学习率，然后在后期缓慢下降。
- 技术原理：  
1.CBAM注意力机制：包含两个子模块  
  (1)通道注意力模块：使用全局平均池化和最大池化来获取通道重要性  
  (2)空间注意力模块：在通道维度上应用池化操作来获取空间重要性
2.OneCycleLR策略：允许学习率先快速上升再缓慢下降，有助于模型快速收敛并避免陷入局部最小值

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
Model Parameters: Total=2,934,066, Trainable=2,934,066
GPU available: False, used: False
TPU available: False, using: 0 TPU cores

Starting training...
Max Epochs: 100
Target Accuracy: 93%+
============================================================
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:881: Checkpoint directory E:\python_exercises\zms_cifar10_cnn\checkpoints exists and is not empty.

   | Name      | Type               | Params | Mode  | FLOPs
------------------------------------------------------------------
0  | conv1     | Conv2d             | 2.6 K  | train | 0    
1  | bn1       | BatchNorm2d        | 192    | train | 0    
2  | cbam1     | CBAM               | 1.2 K  | train | 0    
3  | conv2     | Conv2d             | 165 K  | train | 0    
4  | bn2       | BatchNorm2d        | 384    | train | 0    
5  | cbam2     | CBAM               | 4.7 K  | train | 0    
6  | conv3     | Conv2d             | 663 K  | train | 0    
7  | bn3       | BatchNorm2d        | 768    | train | 0    
8  | cbam3     | CBAM               | 18.5 K | train | 0    
9  | conv4     | Conv2d             | 1.8 M  | train | 0    
10 | bn4       | BatchNorm2d        | 1.0 K  | train | 0    
11 | cbam4     | CBAM               | 32.9 K | train | 0    
12 | gap       | AdaptiveAvgPool2d  | 0      | train | 0    
13 | fc1       | Linear             | 196 K  | train | 0    
14 | dropout1  | Dropout            | 0      | train | 0    
15 | fc2       | Linear             | 73.9 K | train | 0    
16 | dropout2  | Dropout            | 0      | train | 0    
17 | fc3       | Linear             | 1.9 K  | train | 0    
18 | criterion | CrossEntropyLoss   | 0      | train | 0    
19 | train_acc | MulticlassAccuracy | 0      | train | 0    
20 | val_acc   | MulticlassAccuracy | 0      | train | 0    
21 | test_acc  | MulticlassAccuracy | 0      | train | 0    
------------------------------------------------------------------
2.9 M     Trainable params
0         Non-trainable params
2.9 M     Total params
11.736    Total estimated model params size (MB)
54        Modules in train mode
0         Modules in eval mode
0         Total Flops
Sanity Checking: |                                                                               | 0/? [00:00<?, ?it/s]E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:429: Consider setting `persistent_workers=True` in 'val_dataloader' to speed up the dataloader worker initialization.
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\torch\utils\data\dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:429: Consider setting `persistent_workers=True` in 'train_dataloader' to speed up the dataloader worker initialization.
Epoch 0: 100%|█| 782/782 [03:16<00:00,  3.98it/s, v_num=5, train_loss=1.490, train_acc=0.375, val_loss=1.560, val_accurMetric val_accuracy improved. New best score: 0.502                                                                     
Epoch 0, global step 782: 'val_accuracy' reached 0.50160 (best 0.50160), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=00-val_accuracy=0.5016.ckpt' as top 3
Epoch 1: 100%|█| 782/782 [03:27<00:00,  3.77it/s, v_num=5, train_loss=1.270, train_acc=0.688, val_loss=1.400, val_accurMetric val_accuracy improved by 0.096 >= min_delta = 0.0005. New best score: 0.597                                      
Epoch 1, global step 1564: 'val_accuracy' reached 0.59730 (best 0.59730), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=01-val_accuracy=0.5973.ckpt' as top 3
Epoch 2: 100%|█| 782/782 [03:29<00:00,  3.73it/s, v_num=5, train_loss=1.550, train_acc=0.562, val_loss=1.240, val_accurMetric val_accuracy improved by 0.067 >= min_delta = 0.0005. New best score: 0.664                                      
Epoch 2, global step 2346: 'val_accuracy' reached 0.66410 (best 0.66410), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=02-val_accuracy=0.6641.ckpt' as top 3
Epoch 3: 100%|█| 782/782 [03:31<00:00,  3.69it/s, v_num=5, train_loss=1.450, train_acc=0.500, val_loss=1.440, val_accurEpoch 3, global step 3128: 'val_accuracy' reached 0.57930 (best 0.66410), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=03-val_accuracy=0.5793.ckpt' as top 3
Epoch 4: 100%|█| 782/782 [03:28<00:00,  3.75it/s, v_num=5, train_loss=1.410, train_acc=0.688, val_loss=1.160, val_accurMetric val_accuracy improved by 0.045 >= min_delta = 0.0005. New best score: 0.709                                      
Epoch 4, global step 3910: 'val_accuracy' reached 0.70930 (best 0.70930), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=04-val_accuracy=0.7093.ckpt' as top 3
Epoch 5: 100%|█| 782/782 [03:29<00:00,  3.73it/s, v_num=5, train_loss=1.440, train_acc=0.562, val_loss=1.320, val_accurEpoch 5, global step 4692: 'val_accuracy' reached 0.63510 (best 0.70930), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=05-val_accuracy=0.6351.ckpt' as top 3
Epoch 6: 100%|█| 782/782 [03:31<00:00,  3.70it/s, v_num=5, train_loss=1.240, train_acc=0.562, val_loss=1.150, val_accurEpoch 6, global step 5474: 'val_accuracy' reached 0.70940 (best 0.70940), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=06-val_accuracy=0.7094.ckpt' as top 3
Epoch 7: 100%|█| 782/782 [03:36<00:00,  3.61it/s, v_num=5, train_loss=1.440, train_acc=0.688, val_loss=1.140, val_accurMetric val_accuracy improved by 0.005 >= min_delta = 0.0005. New best score: 0.715                                      
Epoch 7, global step 6256: 'val_accuracy' reached 0.71470 (best 0.71470), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=07-val_accuracy=0.7147.ckpt' as top 3
Epoch 8: 100%|█| 782/782 [03:36<00:00,  3.61it/s, v_num=5, train_loss=0.975, train_acc=0.812, val_loss=1.080, val_accurMetric val_accuracy improved by 0.037 >= min_delta = 0.0005. New best score: 0.752                                      
Epoch 8, global step 7038: 'val_accuracy' reached 0.75190 (best 0.75190), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=08-val_accuracy=0.7519.ckpt' as top 3
Epoch 9: 100%|█| 782/782 [03:29<00:00,  3.74it/s, v_num=5, train_loss=1.420, train_acc=0.625, val_loss=1.050, val_accurMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.754                                      
Epoch 9, global step 7820: 'val_accuracy' reached 0.75440 (best 0.75440), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=09-val_accuracy=0.7544.ckpt' as top 3
Epoch 10: 100%|█| 782/782 [03:32<00:00,  3.68it/s, v_num=5, train_loss=1.090, train_acc=0.750, val_loss=1.030, val_accuMetric val_accuracy improved by 0.017 >= min_delta = 0.0005. New best score: 0.771                                      
Epoch 10, global step 8602: 'val_accuracy' reached 0.77140 (best 0.77140), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=10-val_accuracy=0.7714.ckpt' as top 3
Epoch 11: 100%|█| 782/782 [03:28<00:00,  3.75it/s, v_num=5, train_loss=0.928, train_acc=0.750, val_loss=1.000, val_accuMetric val_accuracy improved by 0.007 >= min_delta = 0.0005. New best score: 0.779                                      
Epoch 11, global step 9384: 'val_accuracy' reached 0.77860 (best 0.77860), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=11-val_accuracy=0.7786.ckpt' as top 3
Epoch 12: 100%|█| 782/782 [03:29<00:00,  3.74it/s, v_num=5, train_loss=1.180, train_acc=0.688, val_loss=1.050, val_accuEpoch 12, global step 10166: 'val_accuracy' reached 0.76240 (best 0.77860), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=12-val_accuracy=0.7624.ckpt' as top 3
Epoch 13: 100%|█| 782/782 [03:28<00:00,  3.75it/s, v_num=5, train_loss=1.360, train_acc=0.688, val_loss=1.020, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.780                                      
Epoch 13, global step 10948: 'val_accuracy' reached 0.77980 (best 0.77980), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=13-val_accuracy=0.7798.ckpt' as top 3
Epoch 14: 100%|█| 782/782 [03:29<00:00,  3.74it/s, v_num=5, train_loss=0.855, train_acc=0.875, val_loss=0.994, val_accuMetric val_accuracy improved by 0.012 >= min_delta = 0.0005. New best score: 0.792                                      
Epoch 14, global step 11730: 'val_accuracy' reached 0.79220 (best 0.79220), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=14-val_accuracy=0.7922.ckpt' as top 3
Epoch 15: 100%|█| 782/782 [03:33<00:00,  3.67it/s, v_num=5, train_loss=1.030, train_acc=0.688, val_loss=1.040, val_accuEpoch 15, global step 12512: 'val_accuracy' was not in top 3                                                            
Epoch 16: 100%|█| 782/782 [03:36<00:00,  3.60it/s, v_num=5, train_loss=1.260, train_acc=0.750, val_loss=1.000, val_accuEpoch 16, global step 13294: 'val_accuracy' reached 0.78310 (best 0.79220), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=16-val_accuracy=0.7831.ckpt' as top 3
Epoch 17: 100%|█| 782/782 [03:37<00:00,  3.60it/s, v_num=5, train_loss=1.350, train_acc=0.625, val_loss=1.020, val_accuEpoch 17, global step 14076: 'val_accuracy' was not in top 3                                                            
Epoch 18: 100%|█| 782/782 [03:39<00:00,  3.56it/s, v_num=5, train_loss=1.510, train_acc=0.500, val_loss=1.070, val_accuEpoch 18, global step 14858: 'val_accuracy' was not in top 3                                                            
Epoch 19: 100%|█| 782/782 [03:31<00:00,  3.70it/s, v_num=5, train_loss=0.884, train_acc=0.812, val_loss=1.010, val_accuEpoch 19, global step 15640: 'val_accuracy' reached 0.78160 (best 0.79220), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=19-val_accuracy=0.7816.ckpt' as top 3
Epoch 20: 100%|█| 782/782 [03:36<00:00,  3.60it/s, v_num=5, train_loss=1.090, train_acc=0.812, val_loss=0.976, val_accuMetric val_accuracy improved by 0.004 >= min_delta = 0.0005. New best score: 0.797                                      
Epoch 20, global step 16422: 'val_accuracy' reached 0.79670 (best 0.79670), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=20-val_accuracy=0.7967.ckpt' as top 3
Epoch 21: 100%|█| 782/782 [03:34<00:00,  3.64it/s, v_num=5, train_loss=1.120, train_acc=0.625, val_loss=1.070, val_accuEpoch 21, global step 17204: 'val_accuracy' was not in top 3                                                            
Epoch 22: 100%|█| 782/782 [03:40<00:00,  3.55it/s, v_num=5, train_loss=0.966, train_acc=0.750, val_loss=1.040, val_accuEpoch 22, global step 17986: 'val_accuracy' was not in top 3                                                            
Epoch 23: 100%|█| 782/782 [03:42<00:00,  3.52it/s, v_num=5, train_loss=1.400, train_acc=0.625, val_loss=0.974, val_accuMetric val_accuracy improved by 0.003 >= min_delta = 0.0005. New best score: 0.800                                      
Epoch 23, global step 18768: 'val_accuracy' reached 0.79970 (best 0.79970), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=23-val_accuracy=0.7997.ckpt' as top 3
Epoch 24: 100%|█| 782/782 [04:18<00:00,  3.02it/s, v_num=5, train_loss=1.350, train_acc=0.688, val_loss=1.110, val_accuMetric val_accuracy improved by 0.005 >= min_delta = 0.0005. New best score: 0.805                                      
Epoch 24, global step 19550: 'val_accuracy' reached 0.80460 (best 0.80460), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=24-val_accuracy=0.8046.ckpt' as top 3
Epoch 25: 100%|█| 782/782 [05:13<00:00,  2.49it/s, v_num=5, train_loss=0.885, train_acc=0.875, val_loss=0.945, val_accuMetric val_accuracy improved by 0.008 >= min_delta = 0.0005. New best score: 0.813                                      
Epoch 25, global step 20332: 'val_accuracy' reached 0.81290 (best 0.81290), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=25-val_accuracy=0.8129.ckpt' as top 3
Epoch 26: 100%|█| 782/782 [05:19<00:00,  2.45it/s, v_num=5, train_loss=1.090, train_acc=0.750, val_loss=1.060, val_accuEpoch 26, global step 21114: 'val_accuracy' was not in top 3                                                            
Epoch 27: 100%|█| 782/782 [05:20<00:00,  2.44it/s, v_num=5, train_loss=1.150, train_acc=0.625, val_loss=0.971, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.815                                      
Epoch 27, global step 21896: 'val_accuracy' reached 0.81520 (best 0.81520), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=27-val_accuracy=0.8152.ckpt' as top 3
Epoch 28: 100%|█| 782/782 [05:26<00:00,  2.40it/s, v_num=5, train_loss=1.080, train_acc=0.812, val_loss=0.927, val_accuMetric val_accuracy improved by 0.006 >= min_delta = 0.0005. New best score: 0.822                                      
Epoch 28, global step 22678: 'val_accuracy' reached 0.82160 (best 0.82160), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=28-val_accuracy=0.8216.ckpt' as top 3
Epoch 29: 100%|█| 782/782 [05:31<00:00,  2.36it/s, v_num=5, train_loss=1.220, train_acc=0.625, val_loss=0.922, val_accuEpoch 29, global step 23460: 'val_accuracy' reached 0.81840 (best 0.82160), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=29-val_accuracy=0.8184.ckpt' as top 3
Epoch 30: 100%|█| 782/782 [05:41<00:00,  2.29it/s, v_num=5, train_loss=1.190, train_acc=0.750, val_loss=0.918, val_accuMetric val_accuracy improved by 0.003 >= min_delta = 0.0005. New best score: 0.825                                      
Epoch 30, global step 24242: 'val_accuracy' reached 0.82450 (best 0.82450), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=30-val_accuracy=0.8245.ckpt' as top 3
Epoch 31: 100%|█| 782/782 [05:41<00:00,  2.29it/s, v_num=5, train_loss=1.240, train_acc=0.750, val_loss=0.912, val_accuMetric val_accuracy improved by 0.012 >= min_delta = 0.0005. New best score: 0.837                                      
Epoch 31, global step 25024: 'val_accuracy' reached 0.83690 (best 0.83690), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=31-val_accuracy=0.8369.ckpt' as top 3
Epoch 32: 100%|█| 782/782 [05:47<00:00,  2.25it/s, v_num=5, train_loss=1.070, train_acc=0.812, val_loss=0.893, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.839                                      
Epoch 32, global step 25806: 'val_accuracy' reached 0.83870 (best 0.83870), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=32-val_accuracy=0.8387.ckpt' as top 3
Epoch 33: 100%|█| 782/782 [05:36<00:00,  2.32it/s, v_num=5, train_loss=1.550, train_acc=0.562, val_loss=0.906, val_accuEpoch 33, global step 26588: 'val_accuracy' reached 0.82920 (best 0.83870), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=33-val_accuracy=0.8292.ckpt' as top 3
Epoch 34: 100%|█| 782/782 [05:35<00:00,  2.33it/s, v_num=5, train_loss=0.836, train_acc=0.812, val_loss=0.912, val_accuEpoch 34, global step 27370: 'val_accuracy' was not in top 3                                                            
Epoch 35: 100%|█| 782/782 [05:16<00:00,  2.47it/s, v_num=5, train_loss=0.870, train_acc=0.938, val_loss=1.040, val_accuEpoch 35, global step 28152: 'val_accuracy' was not in top 3                                                            
Epoch 36: 100%|█| 782/782 [05:41<00:00,  2.29it/s, v_num=5, train_loss=0.868, train_acc=0.875, val_loss=0.949, val_accuEpoch 36, global step 28934: 'val_accuracy' was not in top 3                                                            
Epoch 37: 100%|█| 782/782 [05:40<00:00,  2.30it/s, v_num=5, train_loss=1.500, train_acc=0.625, val_loss=1.000, val_accuEpoch 37, global step 29716: 'val_accuracy' was not in top 3                                                            
Epoch 38: 100%|█| 782/782 [05:47<00:00,  2.25it/s, v_num=5, train_loss=1.360, train_acc=0.562, val_loss=0.867, val_accuMetric val_accuracy improved by 0.008 >= min_delta = 0.0005. New best score: 0.846                                      
Epoch 38, global step 30498: 'val_accuracy' reached 0.84650 (best 0.84650), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=38-val_accuracy=0.8465.ckpt' as top 3
Epoch 39: 100%|█| 782/782 [05:46<00:00,  2.26it/s, v_num=5, train_loss=0.842, train_acc=0.938, val_loss=0.872, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.848                                      
Epoch 39, global step 31280: 'val_accuracy' reached 0.84820 (best 0.84820), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=39-val_accuracy=0.8482.ckpt' as top 3
Epoch 40: 100%|█| 782/782 [05:46<00:00,  2.25it/s, v_num=5, train_loss=1.340, train_acc=0.688, val_loss=0.900, val_accuEpoch 40, global step 32062: 'val_accuracy' reached 0.84090 (best 0.84820), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=40-val_accuracy=0.8409.ckpt' as top 3
Epoch 41: 100%|█| 782/782 [05:41<00:00,  2.29it/s, v_num=5, train_loss=0.874, train_acc=0.875, val_loss=0.836, val_accuMetric val_accuracy improved by 0.009 >= min_delta = 0.0005. New best score: 0.857                                      
Epoch 41, global step 32844: 'val_accuracy' reached 0.85700 (best 0.85700), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=41-val_accuracy=0.8570.ckpt' as top 3
Epoch 42: 100%|█| 782/782 [05:58<00:00,  2.18it/s, v_num=5, train_loss=1.130, train_acc=0.750, val_loss=0.853, val_accuEpoch 42, global step 33626: 'val_accuracy' reached 0.85550 (best 0.85700), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=42-val_accuracy=0.8555.ckpt' as top 3
Epoch 43: 100%|█| 782/782 [05:56<00:00,  2.19it/s, v_num=5, train_loss=1.080, train_acc=0.688, val_loss=0.878, val_accuEpoch 43, global step 34408: 'val_accuracy' was not in top 3                                                            
Epoch 44: 100%|█| 782/782 [05:46<00:00,  2.25it/s, v_num=5, train_loss=0.880, train_acc=0.812, val_loss=0.863, val_accuEpoch 44, global step 35190: 'val_accuracy' was not in top 3                                                            
Epoch 45: 100%|█| 782/782 [05:47<00:00,  2.25it/s, v_num=5, train_loss=0.926, train_acc=0.875, val_loss=0.875, val_accuEpoch 45, global step 35972: 'val_accuracy' was not in top 3                                                            
Epoch 46: 100%|█| 782/782 [06:14<00:00,  2.09it/s, v_num=5, train_loss=0.851, train_acc=0.875, val_loss=0.835, val_accuMetric val_accuracy improved by 0.003 >= min_delta = 0.0005. New best score: 0.860                                      
Epoch 46, global step 36754: 'val_accuracy' reached 0.86010 (best 0.86010), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=46-val_accuracy=0.8601.ckpt' as top 3
Epoch 47: 100%|█| 782/782 [05:52<00:00,  2.22it/s, v_num=5, train_loss=1.110, train_acc=0.688, val_loss=0.857, val_accuEpoch 47, global step 37536: 'val_accuracy' reached 0.85920 (best 0.86010), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=47-val_accuracy=0.8592.ckpt' as top 3
Epoch 48: 100%|█| 782/782 [06:05<00:00,  2.14it/s, v_num=5, train_loss=0.923, train_acc=0.875, val_loss=0.850, val_accuEpoch 48, global step 38318: 'val_accuracy' was not in top 3                                                            
Epoch 49: 100%|█| 782/782 [06:13<00:00,  2.09it/s, v_num=5, train_loss=0.856, train_acc=0.875, val_loss=0.893, val_accuEpoch 49, global step 39100: 'val_accuracy' was not in top 3                                                            
Epoch 50: 100%|█| 782/782 [06:13<00:00,  2.09it/s, v_num=5, train_loss=1.050, train_acc=0.750, val_loss=0.834, val_accuEpoch 50, global step 39882: 'val_accuracy' reached 0.85780 (best 0.86010), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=50-val_accuracy=0.8578.ckpt' as top 3
Epoch 51: 100%|█| 782/782 [06:15<00:00,  2.08it/s, v_num=5, train_loss=0.886, train_acc=0.875, val_loss=0.812, val_accuMetric val_accuracy improved by 0.012 >= min_delta = 0.0005. New best score: 0.872                                      
Epoch 51, global step 40664: 'val_accuracy' reached 0.87180 (best 0.87180), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=51-val_accuracy=0.8718.ckpt' as top 3
Epoch 52: 100%|█| 782/782 [06:01<00:00,  2.16it/s, v_num=5, train_loss=1.120, train_acc=0.750, val_loss=0.814, val_accuEpoch 52, global step 41446: 'val_accuracy' reached 0.86900 (best 0.87180), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=52-val_accuracy=0.8690.ckpt' as top 3
Epoch 53: 100%|█| 782/782 [05:38<00:00,  2.31it/s, v_num=5, train_loss=1.060, train_acc=0.750, val_loss=0.815, val_accuEpoch 53, global step 42228: 'val_accuracy' reached 0.86610 (best 0.87180), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=53-val_accuracy=0.8661.ckpt' as top 3
Epoch 54: 100%|█| 782/782 [06:05<00:00,  2.14it/s, v_num=5, train_loss=0.805, train_acc=0.875, val_loss=0.799, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.874                                      
Epoch 54, global step 43010: 'val_accuracy' reached 0.87360 (best 0.87360), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=54-val_accuracy=0.8736.ckpt' as top 3
Epoch 55: 100%|█| 782/782 [05:35<00:00,  2.33it/s, v_num=5, train_loss=1.080, train_acc=0.812, val_loss=0.797, val_accuMetric val_accuracy improved by 0.004 >= min_delta = 0.0005. New best score: 0.878                                      
Epoch 55, global step 43792: 'val_accuracy' reached 0.87780 (best 0.87780), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=55-val_accuracy=0.8778.ckpt' as top 3
Epoch 56: 100%|█| 782/782 [06:08<00:00,  2.12it/s, v_num=5, train_loss=0.808, train_acc=0.875, val_loss=0.797, val_accuEpoch 56, global step 44574: 'val_accuracy' reached 0.87290 (best 0.87780), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=56-val_accuracy=0.8729.ckpt' as top 3
Epoch 57: 100%|█| 782/782 [06:02<00:00,  2.16it/s, v_num=5, train_loss=0.662, train_acc=0.938, val_loss=0.812, val_accuEpoch 57, global step 45356: 'val_accuracy' reached 0.87320 (best 0.87780), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=57-val_accuracy=0.8732.ckpt' as top 3
Epoch 58: 100%|█| 782/782 [06:19<00:00,  2.06it/s, v_num=5, train_loss=1.160, train_acc=0.688, val_loss=0.896, val_accuEpoch 58, global step 46138: 'val_accuracy' was not in top 3                                                            
Epoch 59: 100%|█| 782/782 [06:29<00:00,  2.01it/s, v_num=5, train_loss=0.928, train_acc=0.812, val_loss=0.813, val_accuEpoch 59, global step 46920: 'val_accuracy' reached 0.87600 (best 0.87780), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=59-val_accuracy=0.8760.ckpt' as top 3
Epoch 60: 100%|█| 782/782 [06:26<00:00,  2.02it/s, v_num=5, train_loss=1.170, train_acc=0.750, val_loss=0.814, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.879                                      
Epoch 60, global step 47702: 'val_accuracy' reached 0.87860 (best 0.87860), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=60-val_accuracy=0.8786.ckpt' as top 3
Epoch 61: 100%|█| 782/782 [06:24<00:00,  2.03it/s, v_num=5, train_loss=1.020, train_acc=0.750, val_loss=0.786, val_accuMetric val_accuracy improved by 0.005 >= min_delta = 0.0005. New best score: 0.884                                      
Epoch 61, global step 48484: 'val_accuracy' reached 0.88380 (best 0.88380), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=61-val_accuracy=0.8838.ckpt' as top 3
Epoch 62: 100%|█| 782/782 [06:00<00:00,  2.17it/s, v_num=5, train_loss=0.967, train_acc=0.812, val_loss=0.783, val_accuEpoch 62, global step 49266: 'val_accuracy' reached 0.88210 (best 0.88380), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=62-val_accuracy=0.8821.ckpt' as top 3
Epoch 63: 100%|█| 782/782 [06:35<00:00,  1.98it/s, v_num=5, train_loss=0.790, train_acc=0.875, val_loss=0.760, val_accuMetric val_accuracy improved by 0.006 >= min_delta = 0.0005. New best score: 0.890                                      
Epoch 63, global step 50048: 'val_accuracy' reached 0.88960 (best 0.88960), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=63-val_accuracy=0.8896.ckpt' as top 3
Epoch 64: 100%|█| 782/782 [06:28<00:00,  2.01it/s, v_num=5, train_loss=0.974, train_acc=0.875, val_loss=0.752, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.891                                      
Epoch 64, global step 50830: 'val_accuracy' reached 0.89150 (best 0.89150), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=64-val_accuracy=0.8915.ckpt' as top 3
Epoch 65: 100%|█| 782/782 [06:29<00:00,  2.01it/s, v_num=5, train_loss=0.994, train_acc=0.812, val_loss=0.750, val_accuMetric val_accuracy improved by 0.006 >= min_delta = 0.0005. New best score: 0.898                                      
Epoch 65, global step 51612: 'val_accuracy' reached 0.89790 (best 0.89790), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=65-val_accuracy=0.8979.ckpt' as top 3
Epoch 66: 100%|█| 782/782 [06:35<00:00,  1.98it/s, v_num=5, train_loss=0.729, train_acc=0.875, val_loss=0.752, val_accuEpoch 66, global step 52394: 'val_accuracy' reached 0.89490 (best 0.89790), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=66-val_accuracy=0.8949.ckpt' as top 3
Epoch 67: 100%|█| 782/782 [06:44<00:00,  1.93it/s, v_num=5, train_loss=0.888, train_acc=0.875, val_loss=0.758, val_accuEpoch 67, global step 53176: 'val_accuracy' reached 0.89300 (best 0.89790), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=67-val_accuracy=0.8930.ckpt' as top 3
Epoch 68: 100%|█| 782/782 [06:25<00:00,  2.03it/s, v_num=5, train_loss=0.959, train_acc=0.812, val_loss=0.754, val_accuEpoch 68, global step 53958: 'val_accuracy' reached 0.89580 (best 0.89790), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=68-val_accuracy=0.8958.ckpt' as top 3
Epoch 69: 100%|█| 782/782 [06:09<00:00,  2.12it/s, v_num=5, train_loss=0.839, train_acc=0.938, val_loss=0.746, val_accuEpoch 69, global step 54740: 'val_accuracy' reached 0.89580 (best 0.89790), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=69-val_accuracy=0.8958.ckpt' as top 3
Epoch 70: 100%|█| 782/782 [06:26<00:00,  2.02it/s, v_num=5, train_loss=0.811, train_acc=0.875, val_loss=0.747, val_accuEpoch 70, global step 55522: 'val_accuracy' reached 0.89770 (best 0.89790), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=70-val_accuracy=0.8977.ckpt' as top 3
Epoch 71: 100%|█| 782/782 [06:05<00:00,  2.14it/s, v_num=5, train_loss=0.832, train_acc=0.812, val_loss=0.747, val_accuEpoch 71, global step 56304: 'val_accuracy' reached 0.89720 (best 0.89790), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=71-val_accuracy=0.8972.ckpt' as top 3
Epoch 72: 100%|█| 782/782 [06:22<00:00,  2.04it/s, v_num=5, train_loss=0.586, train_acc=1.000, val_loss=0.738, val_accuMetric val_accuracy improved by 0.005 >= min_delta = 0.0005. New best score: 0.903                                      
Epoch 72, global step 57086: 'val_accuracy' reached 0.90290 (best 0.90290), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=72-val_accuracy=0.9029.ckpt' as top 3
Epoch 73: 100%|█| 782/782 [06:38<00:00,  1.96it/s, v_num=5, train_loss=1.010, train_acc=0.875, val_loss=0.748, val_accuEpoch 73, global step 57868: 'val_accuracy' reached 0.89960 (best 0.90290), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=73-val_accuracy=0.8996.ckpt' as top 3
Epoch 74: 100%|█| 782/782 [06:49<00:00,  1.91it/s, v_num=5, train_loss=1.210, train_acc=0.750, val_loss=0.729, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.904                                      
Epoch 74, global step 58650: 'val_accuracy' reached 0.90370 (best 0.90370), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=74-val_accuracy=0.9037.ckpt' as top 3
Epoch 75: 100%|█| 782/782 [06:59<00:00,  1.87it/s, v_num=5, train_loss=0.699, train_acc=0.875, val_loss=0.729, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.905                                      
Epoch 75, global step 59432: 'val_accuracy' reached 0.90520 (best 0.90520), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=75-val_accuracy=0.9052.ckpt' as top 3
Epoch 76: 100%|█| 782/782 [06:50<00:00,  1.91it/s, v_num=5, train_loss=0.668, train_acc=0.938, val_loss=0.733, val_accuEpoch 76, global step 60214: 'val_accuracy' was not in top 3                                                            
Epoch 77: 100%|█| 782/782 [06:16<00:00,  2.07it/s, v_num=5, train_loss=0.642, train_acc=0.875, val_loss=0.730, val_accuEpoch 77, global step 60996: 'val_accuracy' reached 0.90390 (best 0.90520), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=77-val_accuracy=0.9039.ckpt' as top 3
Epoch 78: 100%|█| 782/782 [06:48<00:00,  1.92it/s, v_num=5, train_loss=0.835, train_acc=0.812, val_loss=0.727, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.906                                      
Epoch 78, global step 61778: 'val_accuracy' reached 0.90630 (best 0.90630), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=78-val_accuracy=0.9063.ckpt' as top 3
Epoch 79: 100%|█| 782/782 [06:37<00:00,  1.97it/s, v_num=5, train_loss=1.050, train_acc=0.750, val_loss=0.719, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.908                                      
Epoch 79, global step 62560: 'val_accuracy' reached 0.90810 (best 0.90810), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=79-val_accuracy=0.9081.ckpt' as top 3
Epoch 80: 100%|█| 782/782 [06:43<00:00,  1.94it/s, v_num=5, train_loss=0.583, train_acc=1.000, val_loss=0.718, val_accuEpoch 80, global step 63342: 'val_accuracy' reached 0.90780 (best 0.90810), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=80-val_accuracy=0.9078.ckpt' as top 3
Epoch 81: 100%|█| 782/782 [06:39<00:00,  1.96it/s, v_num=5, train_loss=0.853, train_acc=0.750, val_loss=0.718, val_accuEpoch 81, global step 64124: 'val_accuracy' reached 0.90690 (best 0.90810), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=81-val_accuracy=0.9069.ckpt' as top 3
Epoch 82: 100%|█| 782/782 [06:03<00:00,  2.15it/s, v_num=5, train_loss=0.708, train_acc=0.875, val_loss=0.716, val_accuMetric val_accuracy improved by 0.002 >= min_delta = 0.0005. New best score: 0.910                                      
Epoch 82, global step 64906: 'val_accuracy' reached 0.91050 (best 0.91050), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=82-val_accuracy=0.9105.ckpt' as top 3
Epoch 83: 100%|█| 782/782 [06:56<00:00,  1.88it/s, v_num=5, train_loss=0.907, train_acc=0.875, val_loss=0.716, val_accuEpoch 83, global step 65688: 'val_accuracy' was not in top 3                                                            
Epoch 84: 100%|█| 782/782 [06:11<00:00,  2.11it/s, v_num=5, train_loss=0.773, train_acc=0.875, val_loss=0.713, val_accuEpoch 84, global step 66470: 'val_accuracy' reached 0.90910 (best 0.91050), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=84-val_accuracy=0.9091.ckpt' as top 3
Epoch 85: 100%|█| 782/782 [06:45<00:00,  1.93it/s, v_num=5, train_loss=0.608, train_acc=1.000, val_loss=0.715, val_accuEpoch 85, global step 67252: 'val_accuracy' reached 0.91030 (best 0.91050), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=85-val_accuracy=0.9103.ckpt' as top 3
Epoch 86: 100%|█| 782/782 [06:16<00:00,  2.07it/s, v_num=5, train_loss=0.633, train_acc=0.938, val_loss=0.712, val_accuMetric val_accuracy improved by 0.001 >= min_delta = 0.0005. New best score: 0.911                                      
Epoch 86, global step 68034: 'val_accuracy' reached 0.91150 (best 0.91150), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=86-val_accuracy=0.9115.ckpt' as top 3
Epoch 87: 100%|█| 782/782 [06:13<00:00,  2.09it/s, v_num=5, train_loss=0.891, train_acc=0.875, val_loss=0.713, val_accuEpoch 87, global step 68816: 'val_accuracy' was not in top 3                                                            
Epoch 88: 100%|█| 782/782 [06:46<00:00,  1.92it/s, v_num=5, train_loss=0.690, train_acc=0.875, val_loss=0.715, val_accuEpoch 88, global step 69598: 'val_accuracy' was not in top 3                                                            
Epoch 89: 100%|█| 782/782 [06:42<00:00,  1.94it/s, v_num=5, train_loss=0.739, train_acc=0.938, val_loss=0.711, val_accuEpoch 89, global step 70380: 'val_accuracy' reached 0.91040 (best 0.91150), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=89-val_accuracy=0.9104.ckpt' as top 3
Epoch 90: 100%|█| 782/782 [07:26<00:00,  1.75it/s, v_num=5, train_loss=0.615, train_acc=0.938, val_loss=0.709, val_accuEpoch 90, global step 71162: 'val_accuracy' was not in top 3                                                            
Epoch 91: 100%|█| 782/782 [06:33<00:00,  1.99it/s, v_num=5, train_loss=0.597, train_acc=0.938, val_loss=0.710, val_accuEpoch 91, global step 71944: 'val_accuracy' was not in top 3                                                            
Epoch 92: 100%|█| 782/782 [06:06<00:00,  2.13it/s, v_num=5, train_loss=0.675, train_acc=0.938, val_loss=0.706, val_accuEpoch 92, global step 72726: 'val_accuracy' reached 0.91190 (best 0.91190), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=92-val_accuracy=0.9119.ckpt' as top 3
Epoch 93: 100%|█| 782/782 [06:53<00:00,  1.89it/s, v_num=5, train_loss=0.825, train_acc=0.875, val_loss=0.707, val_accuEpoch 93, global step 73508: 'val_accuracy' reached 0.91140 (best 0.91190), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=93-val_accuracy=0.9114.ckpt' as top 3
Epoch 94: 100%|█| 782/782 [07:57<00:00,  1.64it/s, v_num=5, train_loss=0.835, train_acc=0.875, val_loss=0.707, val_accuEpoch 94, global step 74290: 'val_accuracy' was not in top 3                                                            
Epoch 95: 100%|█| 782/782 [07:34<00:00,  1.72it/s, v_num=5, train_loss=0.654, train_acc=0.938, val_loss=0.708, val_accuEpoch 95, global step 75072: 'val_accuracy' was not in top 3                                                            
Epoch 96: 100%|█| 782/782 [07:35<00:00,  1.72it/s, v_num=5, train_loss=0.608, train_acc=1.000, val_loss=0.706, val_accuEpoch 96, global step 75854: 'val_accuracy' reached 0.91200 (best 0.91200), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=96-val_accuracy=0.9120.ckpt' as top 3
Epoch 97: 100%|█| 782/782 [07:38<00:00,  1.71it/s, v_num=5, train_loss=0.792, train_acc=0.875, val_loss=0.706, val_accuEpoch 97, global step 76636: 'val_accuracy' reached 0.91190 (best 0.91200), saving model to 'E:\\python_exercises\\zms_cifar10_cnn\\checkpoints\\cifar10-cnn-epoch=97-val_accuracy=0.9119.ckpt' as top 3
Epoch 98: 100%|█| 782/782 [07:37<00:00,  1.71it/s, v_num=5, train_loss=0.716, train_acc=0.938, val_loss=0.707, val_accuEpoch 98, global step 77418: 'val_accuracy' was not in top 3                                                            
Epoch 99: 100%|█| 782/782 [07:25<00:00,  1.76it/s, v_num=5, train_loss=0.921, train_acc=0.875, val_loss=0.706, val_accuEpoch 99, global step 78200: 'val_accuracy' was not in top 3                                                            
`Trainer.fit` stopped: `max_epochs=100` reached.
Epoch 99: 100%|█| 782/782 [07:25<00:00,  1.76it/s, v_num=5, train_loss=0.921, train_acc=0.875, val_loss=0.706, val_accu

Testing best model...
Loading best model: E:\python_exercises\zms_cifar10_cnn\checkpoints\cifar10-cnn-epoch=96-val_accuracy=0.9120.ckpt
E:\python_exercises\zms_cifar10_cnn\.venv\Lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:429: Consider setting `persistent_workers=True` in 'test_dataloader' to speed up the dataloader worker initialization.
Testing DataLoader 0: 100%|██████████████████████████████████████████████████████████| 157/157 [00:11<00:00, 13.46it/s]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      test_accuracy         0.9120000004768372
        test_loss           0.7059800624847412
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
============================================================
Final Test Accuracy: 0.9120
Target not reached, but close to 91.20%
============================================================
Model saved to: ./checkpoints\final_model.pth

Training completed!

```

## 实验6：再次调整后的完整测试
- 时间：2026年1月18日
- 配置：batch_size=64, max_epochs=120
- 结果：测试准确率 90.40%
- 优化内容：
  - 通道数从 96→192→384→512 增加到 128→256→512→768，通过更多的通道数提供更大的模型容量，能够学习更复杂的特征表示，配合正则化防止过拟合
  - model.py新增ResidualBlock类，并添加res1, res2层，ResNet风格的跳跃连接解决了深层网络的梯度消失问题，允许训练更深的网络而不损失性能
  - 从OneCycleLR改为CosineAnnealingWarmRestarts，通过周期性重启学习率帮助模型跳出局部最优，配合120个epoch的训练周期
  - Dropout率从0.4/0.3增加到0.5/0.4，更强的dropout防止过拟合，配合更大的模型容量
  - 新增transforms.RandomAffine(translate=(0.1, 0.1))，随机平移增强模型对位置变化的鲁棒性
