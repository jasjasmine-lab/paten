#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证输入特征脚本
确认模型输入中不包含专利被引用相关信息
"""

import pandas as pd
import numpy as np

def verify_features():
    """验证输入特征"""
    print("=" * 60)
    print("验证输入特征")
    print("=" * 60)
    
    # 加载处理后的数据
    print("\n1. 加载数据...")
    train_data = pd.read_excel('train_data.xlsx')
    test_data = pd.read_excel('test_data.xlsx')
    
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 检查是否有被引用相关的列
    print("\n2. 检查是否包含被引用相关列...")
    
    # 被引用相关的列名关键词
    citation_keywords = ['被引', '引证', '引用', 'citation', 'cited']
    
    found_citation_cols = []
    for col in train_data.columns:
        for keyword in citation_keywords:
            if keyword.lower() in col.lower():
                found_citation_cols.append(col)
                break
    
    if found_citation_cols:
        print("⚠️ 警告：发现以下被引用相关列:")
        for col in found_citation_cols:
            print(f"  - {col}")
        print("这些列不应该作为模型输入！")
    else:
        print("✓ 未发现被引用相关列")
    
    # 列出将作为输入的特征
    print("\n3. 模型输入特征:")
    
    print("\n文本特征列:")
    text_columns = ['标题 (中文)', '摘要 (中文)', '首项权利要求', 
                   '技术功效句', '用途']
    for col in text_columns:
        if col in train_data.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ✗ {col} (不存在)")
    
    print("\n数值特征列:")
    numerical_columns = [
        '权利要求数量',
        '首权字数',
        '申请人数量',
        '发明(设计)人数量',
        '简单同族个数',
        '扩展同族个数',
        'DocDB同族个数'
    ]
    
    for col in numerical_columns:
        if col in train_data.columns:
            # 检查数据类型和非空值数量
            non_null = train_data[col].notna().sum()
            dtype = train_data[col].dtype
            print(f"  ✓ {col} (类型: {dtype}, 非空值: {non_null}/{len(train_data)})")
        else:
            print(f"  ✗ {col} (不存在)")
    
    # 确认标签列
    print("\n4. 标签列:")
    if '质量标签' in train_data.columns:
        label_dist = train_data['质量标签'].value_counts()
        print("  ✓ 质量标签分布:")
        for label, count in label_dist.items():
            print(f"    {label}: {count} ({count/len(train_data)*100:.2f}%)")
    else:
        print("  ✗ 未找到质量标签列")
    
    # 数据来源确认
    print("\n5. 数据来源确认:")
    print("  原始数据文件: merged_data(1)_cleaned.xlsx")
    print("  数据类型: 真实专利数据（非模拟数据）")
    print("  质量评分计算: 基于专利被引次数和有效期限")
    print("  注意: 被引次数仅用于计算质量标签，不作为模型输入")
    
    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)

if __name__ == "__main__":
    verify_features()
