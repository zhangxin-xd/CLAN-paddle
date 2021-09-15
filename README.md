CLAN-Paddle
===
本项目旨在使用百度PaddlePaddle框架复现2019cvpr(oral)：Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation。

本项目由张鑫和崔明迪共同完成。
## 一、简介
- 原论文：[Category-level Adversaries for Semantics Consistent Domain Adaptation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Luo_Taking_a_Closer_Look_at_Domain_Shift_Category-Level_Adversaries_for_CVPR_2019_paper.pdf)

- pytorch代码：[CLAN](https://github.com/RoyalVane/CLAN)

致敬开源精神，respect！！！

## 二、复现结果

本项目通过从SYNTHIA Dataset到Cityscapes Dataset的迁移，实现在Cityscapes Dataset上的语义分割。

||road |side. |buil.|light |sign|vege. |sky |pers. |rider |car |bus |moto |bicycle |mIoU |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|paddle复现|84.04|40.78|79.03|11.14|6.48|80.12|82.78|56.07|22.32|66.71|25.51|12.89|28.14|45.59|
|原文|81.3|37.0|80.1|16.1|13.7|78.2|81.5|53.4|21.2|73.0|32.9|22.6|30.7|47.8|

<img src=https://github.com/zhangxin-xd/CLAN-paddle/blob/main/frankfurt_000001_001464_leftImg8bit_color.png width=50% />

## 三、环境依赖

- Paddle 2.1.2  
- cuDNN 7.6+

## 四、实现

### 训练

#### 下载数据集和预训练权重
  
- [SYNTHIA Dataset](http://synthia-dataset.net/download-2/)  

- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)  

- [Imagenet上的预训练权重](https://pan.baidu.com/s/1iSNNchygnzhrrGiBOs9Umw)  提取码：6666

文件夹组织如下

```
 ├── data/
│   ├── Cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
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

权重保存在./snapshots目录下

### 测试

```
CUDA_VISIBLE_DEVICES=0 python CLAN_evaluate.py --restore-from  ./snapshots/SYS2Cityscapes/SYS_100000.pdparams --save ./result/SYS2Cityscapes_100000
```

此处提供我们训好的权重，可以直接进行测试，图片结果保存在./result中。  

[权重用于直接测试](https://pan.baidu.com/s/1BaUgB87uQ-bM7g4DdqPoMQ ) 提取码：6666

### 评估

```
CUDA_VISIBLE_DEVICES=0 python CLAN_evaluate.py --restore-from  ./snapshots/SYS2Cityscapes/SYS_100000.paradms --save ./result/SYS2Cityscapes_100000
```

此处提供我们训好的权重，可以直接进行测试  

[权重用于直接测试](https://pan.baidu.com/s/1BaUgB87uQ-bM7g4DdqPoMQ)  

提取码：6666

### 计算IoU

```
python CLAN_iou.py ./data/Cityscapes/gtFine/val result/SYS2Cityscapes_100000
```
注意：最好的权重不一定是最后的权重，所以可以通过运行CLAN_evaluate_bulk.py 和 CLAN_iou_bulk.py 评估每个权重的性能。
```
CUDA_VISIBLE_DEVICES=0 python CLAN_evaluate_bulk.py
```
```
python CLAN_iou_bulk.py
```
结果列在./mIoU_results中的excel中。

对于SYNTHIA dataset原论文中使用19类进行训练，但只衡量13类。

## 五、模型信息

|  信息   |  说明 |
|  ----  |  ----  |
| 作者 | 张鑫 |
| 时间 | 2021.09 |
| 框架版本 | Paddle 2.1.2 |
| 应用场景 | 语义分割 |

