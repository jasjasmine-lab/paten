#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据平衡处理脚本 - 欠采样方法
使三类数据数量均衡
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample

def balance_dataset_undersample(data_path, output_path):
    """
    使用欠采样方法平衡数据集
    
    Args:
        data_path: 输入数据文件路径
        output_path: 输出平衡后的数据文件路径
    """
    # 读取数据
    print(f"读取数据: {data_path}")
    df = pd.read_excel(data_path)
    print(f"原始数据量: {len(df)}")
    
    # 统计各类别数量
    label_counts = df['质量标签'].value_counts().sort_index()
    print("\n原始数据分布:")
    for label, count in label_counts.items():
        print(f"  {label}级: {count}条 ({count/len(df)*100:.2f}%)")
    
    # 找到最少的类别数量
    min_count = label_counts.min()
    print(f"\n最少类别数量: {min_count}")
    print(f"将所有类别欠采样到: {min_count}条")
    
    # 对每个类别进行欠采样
    balanced_dfs = []
    
    for label in ['A', 'B', 'C']:
        label_df = df[df['质量标签'] == label]
        
        if len(label_df) > min_count:
            # 欠采样到min_count
            label_df_downsampled = resample(label_df, 
                                           n_samples=min_count,
                                           replace=False,  # 不放回采样
                                           random_state=42)
            balanced_dfs.append(label_df_downsampled)
            print(f"  {label}级: {len(label_df)} -> {min_count} (欠采样)")
        else:
            # 保持原样
            balanced_dfs.append(label_df)
            print(f"  {label}级: {len(label_df)} -> {len(label_df)} (保持)")
    
    # 合并平衡后的数据
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # 随机打乱数据
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 保存平衡后的数据
    balanced_df.to_excel(output_path, index=False)
    print(f"\n平衡后的数据已保存到: {output_path}")
    print(f"总数据量: {len(balanced_df)}")
    
    # 验证平衡后的分布
    print("\n平衡后的数据分布:")
    balanced_counts = balanced_df['质量标签'].value_counts().sort_index()
    for label, count in balanced_counts.items():
        print(f"  {label}级: {count}条 ({count/len(balanced_df)*100:.2f}%)")
    
    return balanced_df

def main():
    """主函数"""
    print("=" * 60)
    print("数据平衡处理 - 欠采样方法")
    print("=" * 60)
    
    # 对训练集进行平衡处理
    print("\n1. 处理训练集...")
    train_balanced = balance_dataset_undersample(
        'train_data_clean.xlsx',
        'train_data_balanced.xlsx'
    )
    
    # 测试集保持不变
    print("\n2. 测试集保持不变")
    print("   测试集应该反映真实数据分布，不进行平衡处理")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("数据平衡处理完成！")
    print("=" * 60)
    
    # 创建一个总结
    summary = {
        '原始训练集': {
            'A类': len(pd.read_excel('train_data_clean.xlsx')[pd.read_excel('train_data_clean.xlsx')['质量标签'] == 'A']),
            'B类': len(pd.read_excel('train_data_clean.xlsx')[pd.read_excel('train_data_clean.xlsx')['质量标签'] == 'B']),
            'C类': len(pd.read_excel('train_data_clean.xlsx')[pd.read_excel('train_data_clean.xlsx')['质量标签'] == 'C']),
        },
        '平衡后训练集': {
            'A类': len(train_balanced[train_balanced['质量标签'] == 'A']),
            'B类': len(train_balanced[train_balanced['质量标签'] == 'B']),
            'C类': len(train_balanced[train_balanced['质量标签'] == 'C']),
        }
    }
    
    print("\n数据平衡总结:")
    print("-" * 40)
    print(f"原始训练集总量: {sum(summary['原始训练集'].values())}")
    print(f"  A类: {summary['原始训练集']['A类']}")
    print(f"  B类: {summary['原始训练集']['B类']}")
    print(f"  C类: {summary['原始训练集']['C类']}")
    print("-" * 40)
    print(f"平衡后训练集总量: {sum(summary['平衡后训练集'].values())}")
    print(f"  A类: {summary['平衡后训练集']['A类']}")
    print(f"  B类: {summary['平衡后训练集']['B类']}")
    print(f"  C类: {summary['平衡后训练集']['C类']}")
    print("-" * 40)
    
    print("\n下一步:")
    print("  使用 train_data_balanced.xlsx 重新训练模型")
    print("  预期改善B类和C类的识别能力")

if __name__ == "__main__":
    main()
