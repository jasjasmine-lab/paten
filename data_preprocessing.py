#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专利质量分级数据预处理脚本
功能：
1. 读取Excel数据
2. 计算质量评分：专利被引次数/(min(专利到期日，2024年12月31日)-专利公开日期)
3. 根据评分分配质量标签（A/B/C）
4. 删除专利被引次数列
5. 划分训练集和测试集
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """加载并预处理数据"""
    
    print("=" * 60)
    print("开始数据预处理...")
    print("=" * 60)
    
    # 1. 读取数据
    print("\n1. 读取Excel数据...")
    df = pd.read_excel('merged_data(1)_cleaned.xlsx')
    print(f"   数据形状: {df.shape}")
    print(f"   总列数: {len(df.columns)}")
    
    # 2. 检查必要的列是否存在
    print("\n2. 检查关键列...")
    required_cols = ['家族被引证次数', '预估到期日', '公开（公告）日']
    
    for col in required_cols:
        if col in df.columns:
            print(f"   ✓ 找到列: {col}")
        else:
            print(f"   ✗ 缺少列: {col}")
            # 尝试查找类似的列名
            similar = [c for c in df.columns if any(keyword in c for keyword in col.split('（')[0])]
            if similar:
                print(f"     可能的替代列: {similar}")
    
    # 3. 处理日期列
    print("\n3. 处理日期数据...")
    
    # 转换日期格式
    date_cols = ['预估到期日', '公开（公告）日']
    for col in date_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"   ✓ 转换日期列: {col}")
                print(f"     有效日期数: {df[col].notna().sum()}")
                print(f"     缺失值数: {df[col].isna().sum()}")
            except Exception as e:
                print(f"   ✗ 转换失败 {col}: {e}")
    
    # 4. 计算质量评分
    print("\n4. 计算专利质量评分...")
    
    # 设置截止日期
    cutoff_date = pd.Timestamp('2024-12-31')
    
    # 计算有效期限（天数）
    df['有效期限_天数'] = np.nan
    
    if '预估到期日' in df.columns and '公开（公告）日' in df.columns:
        # 取预估到期日和2024-12-31中的较小值
        df['有效截止日期'] = df['预估到期日'].apply(
            lambda x: min(x, cutoff_date) if pd.notna(x) else cutoff_date
        )
        
        # 计算天数差
        df['有效期限_天数'] = (df['有效截止日期'] - df['公开（公告）日']).dt.days
        
        # 过滤掉无效数据（天数小于等于0）
        df.loc[df['有效期限_天数'] <= 0, '有效期限_天数'] = np.nan
        
        print(f"   有效期限计算完成")
        print(f"   有效数据数: {df['有效期限_天数'].notna().sum()}")
        print(f"   平均有效期限: {df['有效期限_天数'].mean():.2f} 天")
    
    # 计算质量评分
    if '家族被引证次数' in df.columns:
        # 将有效期限从天转换为年
        df['有效期限_年'] = df['有效期限_天数'] / 365.25
        
        # 计算质量评分
        df['质量评分'] = df['家族被引证次数'] / df['有效期限_年']
        
        # 处理无穷大和NaN值
        df['质量评分'] = df['质量评分'].replace([np.inf, -np.inf], np.nan)
        
        print(f"   质量评分计算完成")
        print(f"   有效评分数: {df['质量评分'].notna().sum()}")
        
        # 显示评分分布
        valid_scores = df['质量评分'].dropna()
        if len(valid_scores) > 0:
            print(f"\n   质量评分统计:")
            print(f"     最小值: {valid_scores.min():.4f}")
            print(f"     25%分位: {valid_scores.quantile(0.25):.4f}")
            print(f"     中位数: {valid_scores.median():.4f}")
            print(f"     平均值: {valid_scores.mean():.4f}")
            print(f"     75%分位: {valid_scores.quantile(0.75):.4f}")
            print(f"     最大值: {valid_scores.max():.4f}")
    
    # 5. 分配质量标签
    print("\n5. 分配质量标签...")
    
    def assign_quality_label(score):
        """根据评分分配质量标签"""
        if pd.isna(score):
            return None
        elif score > 1.85:
            return 'A'
        elif 0.85 <= score <= 1.85:
            return 'B'
        else:
            return 'C'
    
    df['质量标签'] = df['质量评分'].apply(assign_quality_label)
    
    # 统计标签分布
    label_counts = df['质量标签'].value_counts()
    print("\n   质量标签分布:")
    for label in ['A', 'B', 'C']:
        if label in label_counts.index:
            count = label_counts[label]
            percentage = count / len(df) * 100
            print(f"     {label}级: {count} ({percentage:.2f}%)")
    
    null_count = df['质量标签'].isna().sum()
    if null_count > 0:
        print(f"     无效数据: {null_count} ({null_count/len(df)*100:.2f}%)")
    
    # 6. 删除用于计算的中间列和原始被引次数列
    print("\n6. 清理数据...")
    
    # 保留有质量标签的数据
    df_clean = df[df['质量标签'].notna()].copy()
    print(f"   过滤前数据量: {len(df)}")
    print(f"   过滤后数据量: {len(df_clean)}")
    
    # 删除不需要的列
    cols_to_drop = ['家族被引证次数', '有效期限_天数', '有效期限_年', 
                    '有效截止日期', '质量评分']
    
    for col in cols_to_drop:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])
            print(f"   ✓ 删除列: {col}")
    
    # 7. 保存处理后的数据
    print("\n7. 保存处理后的数据...")
    
    output_file = 'patent_data_with_labels.xlsx'
    df_clean.to_excel(output_file, index=False)
    print(f"   ✓ 数据已保存到: {output_file}")
    
    # 8. 生成数据报告
    print("\n" + "=" * 60)
    print("数据预处理完成!")
    print("=" * 60)
    print(f"\n最终数据集:")
    print(f"  - 样本数: {len(df_clean)}")
    print(f"  - 特征数: {len(df_clean.columns) - 1}")  # 减去标签列
    print(f"  - 标签列: 质量标签")
    
    return df_clean

def split_dataset(df, test_size=0.1, random_state=42):
    """划分训练集和测试集"""
    
    print("\n" + "=" * 60)
    print("划分数据集...")
    print("=" * 60)
    
    from sklearn.model_selection import train_test_split
    
    # 分离特征和标签
    X = df.drop(columns=['质量标签'])
    y = df['质量标签']
    
    # 划分数据集（分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n训练集大小: {len(X_train)} ({(1-test_size)*100:.0f}%)")
    print(f"测试集大小: {len(X_test)} ({test_size*100:.0f}%)")
    
    # 检查标签分布
    print("\n训练集标签分布:")
    train_dist = y_train.value_counts()
    for label in sorted(train_dist.index):
        print(f"  {label}: {train_dist[label]} ({train_dist[label]/len(y_train)*100:.2f}%)")
    
    print("\n测试集标签分布:")
    test_dist = y_test.value_counts()
    for label in sorted(test_dist.index):
        print(f"  {label}: {test_dist[label]} ({test_dist[label]/len(y_test)*100:.2f}%)")
    
    # 保存训练集和测试集
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_excel('train_data.xlsx', index=False)
    test_data.to_excel('test_data.xlsx', index=False)
    
    print("\n✓ 训练集已保存到: train_data.xlsx")
    print("✓ 测试集已保存到: test_data.xlsx")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    try:
        # 执行数据预处理
        df_processed = load_and_preprocess_data()
        
        # 划分数据集
        X_train, X_test, y_train, y_test = split_dataset(df_processed, test_size=0.1)
        
        print("\n" + "=" * 60)
        print("所有预处理步骤完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
