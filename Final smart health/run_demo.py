#!/usr/bin/env python3
"""
智慧医疗 - 医学图像分割项目演示脚本

这个脚本演示了完整的项目运行流程：
1. 数据预处理
2. 模型训练
3. 模型评估
4. 结果可视化
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"正在执行: {description}")
    print(f"命令: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"✅ {description} 完成！耗时: {end_time - start_time:.2f}秒")
        if result.stdout:
            print("输出:")
            print(result.stdout)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败！")
        print(f"错误信息: {e.stderr}")
        return False

def check_dependencies():
    """检查依赖包"""
    print("检查项目依赖...")
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'opencv-python', 
        'Pillow', 'albumentations', 'matplotlib', 'tensorboard',
        'scikit-image', 'scipy', 'tqdm', 'PyYAML', 'seaborn', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def create_directories():
    """创建必要的目录"""
    directories = [
        'data', 'data/raw', 'data/processed', 'data/splits',
        'checkpoints', 'logs', 'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ 目录结构创建完成")

def main():
    """主演示函数"""
    print("🏥 智慧医疗 - 医学图像分割项目演示")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 创建目录
    create_directories()
    
    # 步骤1: 创建示例数据
    print("\n📊 步骤1: 创建示例数据")
    if not run_command(
        "python utils/preprocess.py --create_samples --num_samples 50 --split",
        "创建示例数据并划分数据集"
    ):
        print("❌ 数据创建失败，退出演示")
        return
    
    # 步骤2: 训练模型（使用较少的epoch进行演示）
    print("\n🤖 步骤2: 训练模型")
    
    # 修改配置文件以使用较少的epoch进行演示
    config_content = """# 数据集配置
dataset:
  name: "MoNuSeg"
  image_size: 512
  num_classes: 1
  train_split: 0.7
  val_split: 0.2
  test_split: 0.1

# 模型配置
model:
  name: "TransUNet"
  in_channels: 3
  out_channels: 1
  embed_dim: 768
  patch_size: 16
  num_heads: 12
  num_layers: 12
  mlp_ratio: 4.0
  dropout: 0.1

# 训练配置（演示用，减少epoch数）
training:
  batch_size: 2
  num_epochs: 10
  learning_rate: 0.0001
  weight_decay: 0.0001
  optimizer: "Adam"
  scheduler: "CosineAnnealingLR"
  loss_function: "DiceBCELoss"
  
# 数据增强配置
augmentation:
  horizontal_flip_prob: 0.5
  vertical_flip_prob: 0.5
  rotation_limit: 30
  brightness_limit: 0.2
  contrast_limit: 0.2
  gaussian_noise_var: 0.01
  elastic_transform_alpha: 1
  elastic_transform_sigma: 50

# 路径配置
paths:
  data_dir: "data"
  raw_data_dir: "data/raw"
  processed_data_dir: "data/processed"
  splits_dir: "data/splits"
  checkpoints_dir: "checkpoints"
  logs_dir: "logs"
  results_dir: "results"

# 其他配置
misc:
  seed: 42
  num_workers: 2
  device: "cuda"
  save_freq: 5
  log_freq: 50
"""
    
    with open('configs/config.yaml', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    if not run_command("python train.py", "训练TransUNet模型"):
        print("❌ 模型训练失败，但继续演示其他步骤")
    
    # 步骤3: 评估模型
    print("\n📈 步骤3: 评估模型")
    if not run_command("python evaluate.py", "评估模型性能"):
        print("❌ 模型评估失败")
    
    # 步骤4: 可视化结果
    print("\n🎨 步骤4: 可视化结果")
    if not run_command("python visualize.py", "生成可视化结果"):
        print("❌ 可视化失败")
    
    # 显示结果
    print("\n" + "="*60)
    print("🎉 演示完成！")
    print("="*60)
    
    print("\n📁 生成的文件:")
    print("  - checkpoints/: 模型检查点")
    print("  - logs/: TensorBoard日志")
    print("  - results/: 评估结果和可视化")
    print("  - data/: 数据集")
    
    print("\n📊 查看结果:")
    print("  - 训练曲线: results/training_curves.png")
    print("  - 样本结果: results/sample_results.png")
    print("  - 评估指标: results/evaluation_results.csv")
    print("  - 综合报告: results/comprehensive_report.html")
    
    print("\n🔍 使用TensorBoard查看训练过程:")
    print("  tensorboard --logdir logs")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查是否有结果文件
    result_files = [
        'results/training_curves.png',
        'results/sample_results.png',
        'results/evaluation_results.csv',
        'results/comprehensive_report.html'
    ]
    
    print("\n📋 结果文件检查:")
    for file_path in result_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (未生成)")

if __name__ == '__main__':
    main() 