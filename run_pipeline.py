#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专利质量分级模型完整流程执行脚本
包括数据准备和模型训练
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def check_dependencies():
    """检查必要的依赖"""
    print("检查依赖...")
    
    required_packages = [
        'pandas',
        'numpy',
        'torch',
        'transformers',
        'scikit-learn',
        'tqdm',
        'openpyxl',
        'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下包: {', '.join(missing_packages)}")
        print("请运行: pip install " + ' '.join(missing_packages))
        return False
    
    print("✓ 所有依赖已安装")
    return True

def check_data_files():
    """检查必要的数据文件"""
    print("\n检查数据文件...")
    
    required_files = ['merged_patents_inner.xlsx']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"\n缺少以下文件: {', '.join(missing_files)}")
        print("请先运行数据合并脚本: python merge_patent_data_by_publication_number.py")
        return False
    
    print("✓ 数据文件已就绪")
    return True

def run_data_preparation():
    """运行数据预处理"""
    print("\n" + "="*80)
    print("步骤 1: 数据预处理")
    print("="*80)
    
    try:
        # 检查是否已有处理好的数据
        if os.path.exists('train_data_prepared.xlsx') and os.path.exists('test_data_prepared.xlsx'):
            response = input("\n已存在预处理数据，是否重新处理？(y/n): ")
            if response.lower() != 'y':
                print("跳过数据预处理步骤")
                return True
        
        # 运行数据预处理脚本
        result = subprocess.run(
            [sys.executable, 'prepare_patent_data.py'],
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print("数据预处理失败")
            return False
        
        print("\n✓ 数据预处理完成")
        return True
        
    except Exception as e:
        print(f"运行数据预处理时出错: {str(e)}")
        return False

def run_model_training(epochs=10, batch_size=2):
    """运行模型训练"""
    print("\n" + "="*80)
    print("步骤 2: 模型训练")
    print("="*80)
    
    try:
        # 创建Python脚本来调整超参数并运行训练
        training_script = f"""
import sys
sys.path.append('.')

# 修改超参数
import roberta_bilstm_crossattention_model_v2 as model_v2

# 设置超参数
model_v2.BATCH_SIZE = {batch_size}
model_v2.EPOCHS = {epochs}

# 运行训练
if __name__ == "__main__":
    model_v2.main()
"""
        
        # 保存临时训练脚本
        with open('temp_train.py', 'w', encoding='utf-8') as f:
            f.write(training_script)
        
        # 运行训练
        result = subprocess.run(
            [sys.executable, 'roberta_bilstm_crossattention_model_v2.py'],
            capture_output=False,
            text=True
        )
        
        # 删除临时文件
        if os.path.exists('temp_train.py'):
            os.remove('temp_train.py')
        
        if result.returncode != 0:
            print("模型训练失败")
            return False
        
        print("\n✓ 模型训练完成")
        return True
        
    except Exception as e:
        print(f"运行模型训练时出错: {str(e)}")
        return False

def check_gpu():
    """检查GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n✓ GPU可用: {gpu_name}")
            print(f"  显存: {gpu_memory:.2f} GB")
            
            # 检查显存是否足够
            if gpu_memory < 8:
                print("  警告: 显存可能不足，建议使用至少8GB显存的GPU")
                print("  如果出现OOM错误，请减小batch_size或max_length")
            
            return True
        else:
            print("\n⚠ GPU不可用，将使用CPU训练（速度会很慢）")
            response = input("是否继续？(y/n): ")
            return response.lower() == 'y'
    except ImportError:
        print("\n⚠ PyTorch未安装")
        return False

def main():
    """主函数"""
    print("="*80)
    print("专利质量分级模型 - 完整流程")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    parser = argparse.ArgumentParser(description='运行专利质量分级模型流程')
    parser.add_argument('--skip-data-prep', action='store_true', 
                       help='跳过数据预处理步骤')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='训练轮数（默认10）')
    parser.add_argument('--batch-size', type=int, default=2, 
                       help='批次大小（默认2）')
    parser.add_argument('--only-prepare', action='store_true',
                       help='只执行数据预处理')
    
    args = parser.parse_args()
    
    # 1. 检查依赖
    if not check_dependencies():
        return
    
    # 2. 检查GPU
    check_gpu()
    
    # 3. 检查数据文件
    if not args.skip_data_prep:
        if not check_data_files():
            return
    
    # 4. 数据预处理
    if not args.skip_data_prep:
        if not run_data_preparation():
            print("\n数据预处理失败，流程终止")
            return
    
    if args.only_prepare:
        print("\n只执行数据预处理，流程完成")
        return
    
    # 5. 模型训练
    print("\n准备开始模型训练...")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  有效批次大小: {args.batch_size * 8} (使用梯度累积)")
    
    response = input("\n确认开始训练？(y/n): ")
    if response.lower() != 'y':
        print("训练取消")
        return
    
    if not run_model_training(args.epochs, args.batch_size):
        print("\n模型训练失败")
        return
    
    # 6. 完成
    print("\n" + "="*80)
    print("流程完成！")
    print("="*80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n生成的文件:")
    output_files = [
        ('train_data_prepared.xlsx', '训练数据'),
        ('test_data_prepared.xlsx', '测试数据'),
        ('best_model_v2.pth', '最佳模型'),
        ('model_results_v2.json', '评估结果'),
    ]
    
    for file, desc in output_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024 / 1024  # MB
            print(f"  ✓ {file} ({desc}) - {size:.2f} MB")
    
    print("\n下一步:")
    print("1. 查看 model_results_v2.json 了解模型性能")
    print("2. 使用 best_model_v2.pth 进行预测")
    print("3. 分析混淆矩阵和分类报告优化模型")

if __name__ == "__main__":
    main()
