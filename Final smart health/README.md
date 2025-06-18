# 智慧医疗 - 医学图像分割项目

## 项目简介
本项目是一个基于深度学习的医学图像分割系统，使用MoNuSeg细胞核分割数据集，实现TransUNet模型进行细胞核的精确分割。项目涵盖了从数据预处理到模型训练、评估的完整流程。

## 数据集信息
- **数据集名称**: MoNuSeg (Multi-Organ Nucleus Segmentation)
- **图像模态**: 光学显微镜图像
- **分辨率**: 原始图像尺寸不固定，预处理后统一为512×512
- **任务类型**: 细胞核分割
- **数据量**: 训练集70%，验证集20%，测试集10%

## 技术栈
- **深度学习框架**: PyTorch
- **模型架构**: TransUNet (Transformer + U-Net)
- **数据处理**: OpenCV, PIL, Albumentations
- **可视化**: Matplotlib, TensorBoard
- **评估指标**: Dice系数, IoU

## 项目结构
```
智慧医疗/
├── data/                 # 数据集目录
│   ├── raw/             # 原始数据
│   ├── processed/       # 预处理后数据
│   └── splits/          # 训练/验证/测试集划分
├── models/              # 模型定义
│   ├── transunet.py     # TransUNet模型
│   └── utils.py         # 模型工具函数
├── utils/               # 工具函数
│   ├── dataset.py       # 数据集类
│   ├── transforms.py    # 数据增强
│   └── metrics.py       # 评估指标
├── configs/             # 配置文件
│   └── config.yaml      # 训练配置
├── train.py             # 训练脚本
├── evaluate.py          # 评估脚本
├── visualize.py         # 可视化脚本
├── requirements.txt     # 依赖包
└── README.md           # 项目说明
```

## 模型特点
- **TransUNet优势**: 
  - 结合Transformer的全局建模能力
  - 保留U-Net的跳跃连接结构
  - 多尺度特征融合
  - 对小病灶具有更好的敏感性

## 安装和运行
1. 安装依赖：`pip install -r requirements.txt`
2. 下载数据集并放置在`data/raw/`目录
3. 运行数据预处理：`python utils/preprocess.py`
4. 开始训练：`python train.py`
5. 评估模型：`python evaluate.py`
6. 可视化结果：`python visualize.py`

## 开发团队
- 学生姓名：[填写姓名]
- 学号：[填写学号]
- 班级：[填写班级]

## 提交日期
[填写提交日期] 