CLAN-Paddle
===
本项目旨在使用百度PaddlePaddle框架复现2019cvpr(oral)：Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation。

本项目由张鑫和崔明迪共同完成。
## 一、简介
- 原论文：[Category-level Adversaries for Semantics Consistent Domain Adaptation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Luo_Taking_a_Closer_Look_at_Domain_Shift_Category-Level_Adversaries_for_CVPR_2019_paper.pdf)

- pytorch代码：[CLAN](https://github.com/RoyalVane/CLAN)

致敬开源精神，respect！！！

## 二、复现结果

本项目实现了GTA5和SYNTHIA Dataset到Cityscapes Dataset，实现在Cityscapes Dataset上的语义分割。

||GTA5 |SYNTHIA |
|  ----  |  ----  |  ----  | 
|paddle复现|42.2|45.59|
|原文|43.16|47.8


<img src=https://github.com/zhangxin-xd/CLAN-paddle/blob/main/frankfurt_000001_001464_leftImg8bit_color.png width=50% />

## 三、环境依赖

- Paddle 2.1.2  
- cuDNN 7.6+

## 四、实现

### 训练

#### 下载数据集和预训练权重
- [GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/)  
- 
- [SYNTHIA Dataset](http://synthia-dataset.net/download-2/)  

- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)  

- [Imagenet上的预训练权重](https://pan.baidu.com/s/1iSNNchygnzhrrGiBOs9Umw) (6666)

文件夹组织如下

```
 ├── data/
│   ├── Cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
│   ├── GTA5/
|   |   ├── images/
|   |   ├── labels/
│   ├── SYNTHIA/ 
|   |   ├── RAND_CITYSCAPES/
│   └── 			
└── model/
│   ├── pretrained.pdparams
```

#### 开始训练

```
CUDA_VISIBLE_DEVICES=0 python CLAN_train.py --snapshot-dir ./snapshots/SYS2Cityscapes
```

权重和训练日志保存在./snapshots目录下

### 测试

```
CUDA_VISIBLE_DEVICES=0 python CLAN_evaluate.py --restore-from  ./snapshots/SYS2Cityscapes/SYS_100000.pdparams --save ./result/SYS2Cityscapes_100000
```

此处提供我们训好的权重，可以直接进行测试，图片结果保存在./result中。  

[权重SYS](https://pan.baidu.com/s/1BaUgB87uQ-bM7g4DdqPoMQ ) (6666)

[权重GAT](https://pan.baidu.com/s/1wp8Lczp_o8v0aMTONPyUtw ) (6666)


### 计算IoU

```
python CLAN_iou.py ./data/Cityscapes/gtFine/val result/SYS2Cityscapes_100000
```
注意：最好的权重不一定是最后的权重，所以可以通过运行CLAN_evaluate_bulk.py 和 CLAN_iou_bulk.py 评估训练中得到的每个权重的性能。
```
CUDA_VISIBLE_DEVICES=0 python CLAN_evaluate_bulk.py
```
```
python CLAN_iou_bulk.py
```
结果列在./mIoU_results中的excel中。

对于GAT5 dataset原论文中使用19类进行训练，衡量19类。
对于SYNTHIA dataset原论文中使用19类进行训练，但只衡量13类。

## 五、模型信息

|  信息   |  说明 |
|  ----  |  ----  |
| 作者 | 张鑫 |
| 时间 | 2021.09 |
| 框架版本 | Paddle 2.1.2 |
| 应用场景 | 语义分割 |

