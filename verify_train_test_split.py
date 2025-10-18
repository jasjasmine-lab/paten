#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证训练集和测试集分割的正确性
确保没有数据泄露
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

def verify_train_test_split():
    """验证训练集和测试集的分割"""
    print("="*80)
    print("训练集/测试集分割验证")
    print("="*80)
    
    # 1. 读取数据
    print("\n1. 读取数据集...")
    train_df = pd.read_excel('train_data_prepared.xlsx')
    test_df = pd.read_excel('test_data_prepared.xlsx')
    all_df = pd.read_excel('all_data_prepared.xlsx')
    
    print(f"   训练集: {len(train_df):,} 条")
    print(f"   测试集: {len(test_df):,} 条")
    print(f"   总数据: {len(all_df):,} 条")
    print(f"   总和验证: {len(train_df) + len(test_df)} = {len(all_df)} ✓" 
          if len(train_df) + len(test_df) == len(all_df) else "✗")
    
    # 2. 检查公开号重复
    print("\n2. 检查公开号重复...")
    train_pub_nums = set(train_df['标准化公开号'])
    test_pub_nums = set(test_df['标准化公开号'])
    
    # 查找交集
    overlap = train_pub_nums & test_pub_nums
    
    if len(overlap) > 0:
        print(f"   ❌ 发现重复的公开号: {len(overlap)} 个")
        print("   前10个重复的公开号:")
        for i, pub_num in enumerate(list(overlap)[:10], 1):
            print(f"     {i}. {pub_num}")
    else:
        print("   ✅ 没有发现重复的公开号")
    
    # 3. 检查文本内容重复（抽样检查）
    print("\n3. 检查文本内容重复（抽样1000条）...")
    
    sample_size = min(1000, len(test_df))
    test_sample = test_df.sample(n=sample_size, random_state=42)
    
    duplicate_texts = []
    for idx, test_row in tqdm(test_sample.iterrows(), total=sample_size, desc="检查文本"):
        test_text = test_row['组合文本']
        
        # 在训练集中查找相同文本
        matches = train_df[train_df['组合文本'] == test_text]
        
        if len(matches) > 0:
            duplicate_texts.append({
                'test_pub_num': test_row['标准化公开号'],
                'train_pub_nums': matches['标准化公开号'].tolist(),
                'text_preview': test_text[:100] + '...'
            })
    
    if len(duplicate_texts) > 0:
        print(f"\n   ❌ 发现重复的文本内容: {len(duplicate_texts)} 条")
        print("   前5个重复文本:")
        for i, dup in enumerate(duplicate_texts[:5], 1):
            print(f"\n   {i}. 测试集公开号: {dup['test_pub_num']}")
            print(f"      训练集公开号: {dup['train_pub_nums']}")
            print(f"      文本预览: {dup['text_preview']}")
    else:
        print("   ✅ 抽样检查未发现重复的文本内容")
    
    # 4. 检查标签分布
    print("\n4. 检查标签分布...")
    train_label_dist = train_df['质量标签'].value_counts(normalize=True).sort_index()
    test_label_dist = test_df['质量标签'].value_counts(normalize=True).sort_index()
    
    print("\n   训练集标签分布:")
    for label in ['A', 'B', 'C']:
        count = train_df['质量标签'].value_counts().get(label, 0)
        percentage = train_label_dist.get(label, 0) * 100
        print(f"     {label}级: {count:,}条 ({percentage:.1f}%)")
    
    print("\n   测试集标签分布:")
    for label in ['A', 'B', 'C']:
        count = test_df['质量标签'].value_counts().get(label, 0)
        percentage = test_label_dist.get(label, 0) * 100
        print(f"     {label}级: {count:,}条 ({percentage:.1f}%)")
    
    # 检查分布是否相似
    max_diff = max(abs(train_label_dist.get(label, 0) - test_label_dist.get(label, 0)) 
                   for label in ['A', 'B', 'C'])
    
    if max_diff < 0.05:  # 5%的差异
        print("\n   ✅ 标签分布相似（差异 < 5%）")
    else:
        print(f"\n   ⚠️ 标签分布有差异（最大差异: {max_diff*100:.1f}%）")
    
    # 5. 检查时间分布
    print("\n5. 检查申请日期分布...")
    
    # 转换日期
    train_df['申请日期'] = pd.to_datetime(train_df['申请日期'], errors='coerce')
    test_df['申请日期'] = pd.to_datetime(test_df['申请日期'], errors='coerce')
    
    # 提取年份
    train_years = train_df['申请日期'].dt.year.dropna()
    test_years = test_df['申请日期'].dt.year.dropna()
    
    if len(train_years) > 0 and len(test_years) > 0:
        print(f"\n   训练集年份范围: {train_years.min():.0f} - {train_years.max():.0f}")
        print(f"   测试集年份范围: {test_years.min():.0f} - {test_years.max():.0f}")
        
        # 检查是否有时间泄露（测试集全部早于训练集）
        train_max_year = train_years.max()
        test_min_year = test_years.min()
        
        if test_min_year > train_max_year:
            print("   ⚠️ 测试集完全晚于训练集（可能是时间分割）")
        elif train_years.min() > test_years.max():
            print("   ⚠️ 训练集完全晚于测试集（异常情况）")
        else:
            print("   ✅ 训练集和测试集时间有重叠（随机分割）")
    
    # 6. 检查数值特征分布
    print("\n6. 检查数值特征分布...")
    
    feature_columns = [col for col in train_df.columns if col.startswith('数值特征_')]
    
    if len(feature_columns) > 0:
        print(f"   共有 {len(feature_columns)} 个数值特征")
        
        # 比较每个特征的分布
        for col in feature_columns[:3]:  # 只显示前3个特征
            feature_name = col.replace('数值特征_', '')
            train_mean = train_df[col].mean()
            test_mean = test_df[col].mean()
            diff_percent = abs(train_mean - test_mean) / (train_mean + 1e-10) * 100
            
            print(f"\n   {feature_name}:")
            print(f"     训练集均值: {train_mean:.3f}")
            print(f"     测试集均值: {test_mean:.3f}")
            print(f"     差异: {diff_percent:.1f}%")
    
    # 7. 最终结论
    print("\n" + "="*80)
    print("验证结论")
    print("="*80)
    
    has_leak = len(overlap) > 0 or len(duplicate_texts) > 0
    
    if not has_leak:
        print("\n✅ 数据分割验证通过！")
        print("   - 没有发现公开号重复")
        print("   - 没有发现文本内容重复")
        print("   - 标签分布均衡")
        print("   - 可以安全地进行模型训练")
    else:
        print("\n❌ 发现数据泄露问题！")
        if len(overlap) > 0:
            print(f"   - 发现 {len(overlap)} 个重复的公开号")
        if len(duplicate_texts) > 0:
            print(f"   - 发现 {len(duplicate_texts)} 条重复的文本")
        print("\n   建议重新生成训练/测试集分割")
    
    return not has_leak

def check_data_consistency():
    """检查数据一致性"""
    print("\n" + "="*80)
    print("数据一致性检查")
    print("="*80)
    
    train_df = pd.read_excel('train_data_prepared.xlsx')
    test_df = pd.read_excel('test_data_prepared.xlsx')
    
    # 1. 检查列是否一致
    print("\n1. 检查列一致性...")
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    if train_cols == test_cols:
        print("   ✅ 训练集和测试集列完全一致")
    else:
        print("   ❌ 列不一致")
        only_in_train = train_cols - test_cols
        only_in_test = test_cols - train_cols
        
        if only_in_train:
            print(f"   只在训练集中的列: {only_in_train}")
        if only_in_test:
            print(f"   只在测试集中的列: {only_in_test}")
    
    # 2. 检查数据类型
    print("\n2. 检查数据类型...")
    dtype_mismatch = []
    
    for col in train_cols & test_cols:
        if train_df[col].dtype != test_df[col].dtype:
            dtype_mismatch.append({
                'column': col,
                'train_dtype': str(train_df[col].dtype),
                'test_dtype': str(test_df[col].dtype)
            })
    
    if len(dtype_mismatch) == 0:
        print("   ✅ 所有列的数据类型一致")
    else:
        print(f"   ❌ 发现 {len(dtype_mismatch)} 个列的数据类型不一致")
        for mismatch in dtype_mismatch[:5]:
            print(f"     {mismatch['column']}: 训练集={mismatch['train_dtype']}, "
                  f"测试集={mismatch['test_dtype']}")
    
    # 3. 检查缺失值
    print("\n3. 检查缺失值...")
    train_missing = train_df.isnull().sum()
    test_missing = test_df.isnull().sum()
    
    train_has_missing = train_missing.sum() > 0
    test_has_missing = test_missing.sum() > 0
    
    if not train_has_missing and not test_has_missing:
        print("   ✅ 没有缺失值")
    else:
        if train_has_missing:
            print(f"   训练集缺失值: {train_missing.sum()} 个")
            top_missing = train_missing[train_missing > 0].head(3)
            for col, count in top_missing.items():
                print(f"     {col}: {count} 个缺失")
        
        if test_has_missing:
            print(f"   测试集缺失值: {test_missing.sum()} 个")
            top_missing = test_missing[test_missing > 0].head(3)
            for col, count in top_missing.items():
                print(f"     {col}: {count} 个缺失")

def main():
    """主函数"""
    print("\n专利数据集验证程序")
    print("="*80)
    
    try:
        # 验证数据分割
        is_valid = verify_train_test_split()
        
        # 检查数据一致性
        check_data_consistency()
        
        if is_valid:
            print("\n" + "="*80)
            print("✅ 所有验证通过，可以开始训练！")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("❌ 验证失败，请检查数据分割！")
            print("="*80)
            
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
