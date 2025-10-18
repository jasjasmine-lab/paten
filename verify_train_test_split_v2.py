#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证训练集和测试集分割的正确性（修复版）
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
        print(f"\n   ⚠️ 发现重复的文本内容: {len(duplicate_texts)} 条")
        print("   前5个重复文本:")
        for i, dup in enumerate(duplicate_texts[:5], 1):
            print(f"\n   {i}. 测试集公开号: {dup['test_pub_num']}")
            print(f"      训练集公开号: {dup['train_pub_nums']}")
            print(f"      文本预览: {dup['text_preview']}")
        
        # 分析重复原因
        print("\n   重复文本分析:")
        print(f"     抽样测试: {sample_size} 条")
        print(f"     重复数量: {len(duplicate_texts)} 条")
        print(f"     重复比例: {len(duplicate_texts)/sample_size*100:.2f}%")
        
        if len(duplicate_texts)/sample_size < 0.02:  # 小于2%
            print("     结论: 重复比例较低（<2%），可能是同族专利或相似专利")
            print("     建议: 可以继续训练，但需要注意评估时的解释")
        else:
            print("     结论: 重复比例较高，可能影响模型评估的可靠性")
            print("     建议: 考虑重新分割数据集")
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
    
    # 5. 检查数值特征分布
    print("\n5. 检查数值特征分布...")
    
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
    
    # 6. 检查可用列
    print("\n6. 检查数据列...")
    print(f"   训练集列数: {len(train_df.columns)}")
    print(f"   测试集列数: {len(test_df.columns)}")
    
    # 显示前几个列名
    print("\n   主要列名:")
    for col in list(train_df.columns)[:10]:
        print(f"     - {col}")
    
    # 7. 最终结论
    print("\n" + "="*80)
    print("验证结论")
    print("="*80)
    
    has_serious_leak = len(overlap) > 0
    has_minor_leak = len(duplicate_texts) > 0 and len(duplicate_texts)/sample_size >= 0.02
    
    if not has_serious_leak and not has_minor_leak:
        print("\n✅ 数据分割验证通过！")
        print("   - 没有发现公开号重复")
        print("   - 文本重复比例很低或没有")
        print("   - 标签分布均衡")
        print("   - 可以安全地进行模型训练")
        return True
    elif not has_serious_leak and len(duplicate_texts) > 0 and len(duplicate_texts)/sample_size < 0.02:
        print("\n⚠️ 发现少量文本重复，但可以接受")
        print(f"   - 重复比例: {len(duplicate_texts)/sample_size*100:.2f}%")
        print("   - 可能是同族专利或相似专利")
        print("   - 建议继续训练，但在评估时注意解释")
        return True
    else:
        print("\n❌ 发现数据泄露问题！")
        if len(overlap) > 0:
            print(f"   - 发现 {len(overlap)} 个重复的公开号")
        if has_minor_leak:
            print(f"   - 文本重复比例过高: {len(duplicate_texts)/sample_size*100:.2f}%")
        print("\n   建议重新生成训练/测试集分割")
        return False

def analyze_duplicate_texts():
    """深入分析重复文本"""
    print("\n" + "="*80)
    print("重复文本深入分析")
    print("="*80)
    
    train_df = pd.read_excel('train_data_prepared.xlsx')
    test_df = pd.read_excel('test_data_prepared.xlsx')
    
    print("\n查找所有重复文本（可能需要较长时间）...")
    
    # 找出所有重复的文本
    all_duplicates = []
    batch_size = 100
    
    for i in tqdm(range(0, len(test_df), batch_size), desc="批次处理"):
        batch = test_df.iloc[i:i+batch_size]
        for idx, test_row in batch.iterrows():
            test_text = test_row['组合文本']
            matches = train_df[train_df['组合文本'] == test_text]
            
            if len(matches) > 0:
                all_duplicates.append({
                    'test_pub_num': test_row['标准化公开号'],
                    'train_pub_nums': matches['标准化公开号'].tolist(),
                    'test_label': test_row['质量标签'],
                    'train_labels': matches['质量标签'].tolist()
                })
    
    if len(all_duplicates) > 0:
        print(f"\n发现总重复数: {len(all_duplicates)} 条")
        print(f"重复比例: {len(all_duplicates)/len(test_df)*100:.2f}%")
        
        # 分析标签一致性
        label_consistent = 0
        label_inconsistent = 0
        
        for dup in all_duplicates:
            if all(label == dup['test_label'] for label in dup['train_labels']):
                label_consistent += 1
            else:
                label_inconsistent += 1
        
        print(f"\n标签一致性分析:")
        print(f"  标签一致: {label_consistent} 条 ({label_consistent/len(all_duplicates)*100:.1f}%)")
        print(f"  标签不一致: {label_inconsistent} 条 ({label_inconsistent/len(all_duplicates)*100:.1f}%)")
        
        # 分析公开号模式
        print(f"\n公开号模式分析:")
        test_patterns = {}
        train_patterns = {}
        
        for dup in all_duplicates:
            # 测试集公开号前缀
            test_prefix = dup['test_pub_num'][:2]
            test_patterns[test_prefix] = test_patterns.get(test_prefix, 0) + 1
            
            # 训练集公开号前缀
            for train_pub in dup['train_pub_nums']:
                train_prefix = train_pub[:2]
                train_patterns[train_prefix] = train_patterns.get(train_prefix, 0) + 1
        
        print("\n  测试集公开号前缀分布:")
        for prefix, count in sorted(test_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {prefix}: {count} 条")
        
        print("\n  训练集公开号前缀分布:")
        for prefix, count in sorted(train_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {prefix}: {count} 条")
        
        # 保存详细分析结果
        duplicate_df = pd.DataFrame(all_duplicates)
        duplicate_df.to_excel('duplicate_analysis.xlsx', index=False)
        print("\n详细重复分析已保存到: duplicate_analysis.xlsx")
    else:
        print("\n✅ 没有发现重复文本")

def main():
    """主函数"""
    print("\n专利数据集验证程序 V2")
    print("="*80)
    
    try:
        # 验证数据分割
        is_valid = verify_train_test_split()
        
        # 如果有重复，进行深入分析
        if not is_valid:
            print("\n是否进行深入分析？这可能需要较长时间。")
            # analyze_duplicate_texts()
        
        if is_valid:
            print("\n" + "="*80)
            print("✅ 验证完成，可以开始训练！")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("⚠️ 发现一些问题，但影响较小，仍可继续训练")
            print("="*80)
            
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
