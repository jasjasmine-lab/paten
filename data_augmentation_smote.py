#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMOTE数据增强脚本 - 混合采样策略
使用SMOTE过采样少数类，轻微欠采样多数类
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def prepare_features_for_smote(df):
    """
    准备用于SMOTE的特征
    提取数值特征用于合成新样本
    """
    # 选择数值特征列（不包含被引用信息）
    numerical_columns = [
        '权利要求数量',
        '首权字数',
        '申请人数量',
        '发明(设计)人数量',
        '简单同族个数',
        '扩展同族个数',
        'DocDB同族个数'
    ]
    
    # 提取存在的数值特征
    features = []
    valid_columns = []
    
    for col in numerical_columns:
        if col in df.columns:
            col_data = pd.to_numeric(df[col], errors='coerce')
            col_data = col_data.fillna(col_data.median() if col_data.notna().any() else 0)
            features.append(col_data.values.reshape(-1, 1))
            valid_columns.append(col)
    
    print(f"使用的数值特征: {valid_columns}")
    
    if features:
        features = np.hstack(features)
        # 标准化
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        # 如果没有数值特征，创建随机特征
        features = np.random.randn(len(df), 5)
        print("警告：未找到数值特征，使用随机特征")
    
    return features

def augment_data_with_smote(data_path, output_path, target_samples_per_class=2000):
    """
    使用SMOTE和混合采样策略增强数据
    
    Args:
        data_path: 输入数据文件路径
        output_path: 输出增强后的数据文件路径
        target_samples_per_class: 每个类别的目标样本数
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
    
    # 准备特征和标签
    X_features = prepare_features_for_smote(df)
    
    # 标签编码
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['质量标签'])
    
    # 设置采样策略
    print(f"\n目标：每类{target_samples_per_class}个样本")
    
    # 计算采样策略
    sampling_strategy = {}
    for i, label in enumerate(label_encoder.classes_):
        current_count = np.sum(y == i)
        if current_count < target_samples_per_class:
            sampling_strategy[i] = target_samples_per_class
        else:
            # 对于多数类，保持90%的样本
            sampling_strategy[i] = int(current_count * 0.9)
    
    print("\n采样策略:")
    for i, label in enumerate(label_encoder.classes_):
        current = np.sum(y == i)
        target = sampling_strategy[i]
        print(f"  {label}级: {current} -> {target}")
    
    # 应用SMOTE过采样
    print("\n应用SMOTE过采样...")
    try:
        # 首先过采样少数类
        smote = SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X_features, y)
        
        # 然后对多数类进行轻微欠采样
        under_sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_final, y_final = under_sampler.fit_resample(X_resampled, y_resampled)
        
    except Exception as e:
        print(f"SMOTE失败: {e}")
        print("使用简单的重采样策略...")
        
        # 备用方案：简单的重采样
        balanced_dfs = []
        for label in label_encoder.classes_:
            label_idx = label_encoder.transform([label])[0]
            label_mask = y == label_idx
            label_df = df[df['质量标签'] == label]
            
            if len(label_df) < target_samples_per_class:
                # 过采样
                label_df_upsampled = resample(label_df, 
                                             n_samples=target_samples_per_class,
                                             replace=True,
                                             random_state=42)
                balanced_dfs.append(label_df_upsampled)
            else:
                # 欠采样
                label_df_downsampled = resample(label_df, 
                                               n_samples=int(len(label_df) * 0.9),
                                               replace=False,
                                               random_state=42)
                balanced_dfs.append(label_df_downsampled)
        
        # 合并数据
        augmented_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # 随机打乱
        augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 保存
        augmented_df.to_excel(output_path, index=False)
        
        print(f"\n增强后的数据已保存到: {output_path}")
        print(f"总数据量: {len(augmented_df)}")
        
        # 验证增强后的分布
        print("\n增强后的数据分布:")
        augmented_counts = augmented_df['质量标签'].value_counts().sort_index()
        for label, count in augmented_counts.items():
            print(f"  {label}级: {count}条 ({count/len(augmented_df)*100:.2f}%)")
        
        return augmented_df
    
    # 如果SMOTE成功，需要将结果映射回原始数据
    print("将SMOTE结果映射回原始数据...")
    
    # 获取原始索引
    original_indices = np.arange(len(df))
    
    # 创建增强后的数据框
    augmented_indices = []
    for i in range(len(y_final)):
        if i < len(original_indices):
            augmented_indices.append(original_indices[i])
        else:
            # 对于合成的样本，随机选择一个同类样本作为模板
            same_class_mask = y == y_final[i]
            same_class_indices = original_indices[same_class_mask]
            template_idx = np.random.choice(same_class_indices)
            augmented_indices.append(template_idx)
    
    # 根据索引创建增强数据
    augmented_df = df.iloc[augmented_indices].copy()
    augmented_df = augmented_df.reset_index(drop=True)
    
    # 随机打乱数据
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 保存增强后的数据
    augmented_df.to_excel(output_path, index=False)
    print(f"\n增强后的数据已保存到: {output_path}")
    print(f"总数据量: {len(augmented_df)}")
    
    # 验证增强后的分布
    print("\n增强后的数据分布:")
    augmented_counts = augmented_df['质量标签'].value_counts().sort_index()
    for label, count in augmented_counts.items():
        print(f"  {label}级: {count}条 ({count/len(augmented_df)*100:.2f}%)")
    
    return augmented_df

def main():
    """主函数"""
    print("=" * 60)
    print("SMOTE数据增强 - 混合采样策略")
    print("=" * 60)
    
    # 对训练集进行增强处理
    print("\n1. 处理训练集...")
    
    # 设置每类目标样本数
    target_samples = 2200  # 适中的目标值
    
    train_augmented = augment_data_with_smote(
        'train_data_clean.xlsx',
        'train_data_augmented.xlsx',
        target_samples_per_class=target_samples
    )
    
    # 测试集保持不变
    print("\n2. 测试集保持不变")
    print("   测试集应该反映真实数据分布，不进行增强处理")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("数据增强处理完成！")
    print("=" * 60)
    
    # 创建总结
    original_df = pd.read_excel('train_data_clean.xlsx')
    summary = {
        '原始训练集': {
            'A类': len(original_df[original_df['质量标签'] == 'A']),
            'B类': len(original_df[original_df['质量标签'] == 'B']),
            'C类': len(original_df[original_df['质量标签'] == 'C']),
        },
        '增强后训练集': {
            'A类': len(train_augmented[train_augmented['质量标签'] == 'A']),
            'B类': len(train_augmented[train_augmented['质量标签'] == 'B']),
            'C类': len(train_augmented[train_augmented['质量标签'] == 'C']),
        }
    }
    
    print("\n数据增强总结:")
    print("-" * 40)
    print(f"原始训练集总量: {sum(summary['原始训练集'].values())}")
    print(f"  A类: {summary['原始训练集']['A类']}")
    print(f"  B类: {summary['原始训练集']['B类']}")
    print(f"  C类: {summary['原始训练集']['C类']}")
    print("-" * 40)
    print(f"增强后训练集总量: {sum(summary['增强后训练集'].values())}")
    print(f"  A类: {summary['增强后训练集']['A类']}")
    print(f"  B类: {summary['增强后训练集']['B类']}")
    print(f"  C类: {summary['增强后训练集']['C类']}")
    print("-" * 40)
    
    print("\n下一步:")
    print("  使用 train_data_augmented.xlsx 训练优化模型")
    print("  预期显著改善模型性能")

if __name__ == "__main__":
    main()
