#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
去除引证相关特征，避免信息泄露
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def remove_citation_features():
    """去除可能导致信息泄露的引证相关特征"""
    print("=" * 60)
    print("去除引证相关特征，避免信息泄露")
    print("=" * 60)
    
    # 加载带标签的数据
    print("\n1. 加载原始数据...")
    df = pd.read_excel('patent_data_with_labels.xlsx')
    print(f"原始数据形状: {df.shape}")
    
    # 需要删除的引证相关列
    citation_columns = [
        '家族引证',
        '家族被引证', 
        '家族引证申请人',
        '家族被引证申请人',
        '家族引证次数',
        '家族被引证次数'
    ]
    
    print("\n2. 检查并删除引证相关列...")
    removed_columns = []
    for col in citation_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
            removed_columns.append(col)
            print(f"  ✓ 已删除: {col}")
    
    if removed_columns:
        print(f"\n共删除 {len(removed_columns)} 个引证相关列")
    else:
        print("\n未发现需要删除的引证相关列")
    
    # 再次检查是否还有其他引证相关列
    print("\n3. 二次检查引证相关列...")
    citation_keywords = ['引证', '引用', '被引', 'citation', 'cited']
    remaining_citation_cols = []
    
    for col in df.columns:
        for keyword in citation_keywords:
            if keyword.lower() in col.lower():
                remaining_citation_cols.append(col)
                break
    
    if remaining_citation_cols:
        print("发现其他可能的引证相关列:")
        for col in remaining_citation_cols:
            print(f"  - {col}")
            if col not in ['质量标签']:
                df = df.drop(columns=[col])
                print(f"    已删除: {col}")
    else:
        print("✓ 未发现其他引证相关列")
    
    print(f"\n4. 处理后数据形状: {df.shape}")
    
    # 保存清理后的数据
    print("\n5. 保存清理后的数据...")
    df.to_excel('patent_data_no_citation.xlsx', index=False)
    print("  ✓ 已保存到: patent_data_no_citation.xlsx")
    
    # 重新划分训练集和测试集
    print("\n6. 重新划分数据集...")
    
    X = df.drop(columns=['质量标签'])
    y = df['质量标签']
    
    # 90%训练，10%测试，分层抽样
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    
    # 合并特征和标签
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # 保存新的训练集和测试集
    train_data.to_excel('train_data_clean.xlsx', index=False)
    test_data.to_excel('test_data_clean.xlsx', index=False)
    
    print(f"  训练集: {len(train_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    print("  ✓ 已保存到: train_data_clean.xlsx, test_data_clean.xlsx")
    
    # 统计信息
    print("\n7. 数据统计:")
    print(f"  特征数量: {len(X.columns)}")
    print("\n训练集标签分布:")
    print(y_train.value_counts())
    print("\n测试集标签分布:")
    print(y_test.value_counts())
    
    print("\n" + "=" * 60)
    print("数据清理完成！")
    print("已去除所有引证相关特征，避免信息泄露")
    print("=" * 60)
    
    return df, train_data, test_data

if __name__ == "__main__":
    remove_citation_features()
