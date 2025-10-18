#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分脚本
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("开始划分数据集...")

# 读取已处理的数据
df = pd.read_excel('patent_data_with_labels.xlsx')
print(f"数据集形状: {df.shape}")

# 检查质量标签分布
print("\n质量标签分布:")
print(df['质量标签'].value_counts())

# 分离特征和标签
X = df.drop(columns=['质量标签'])
y = df['质量标签']

# 划分数据集（90%训练，10%测试，分层抽样）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# 保存训练集和测试集
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

print("\n保存数据集...")
train_data.to_excel('train_data.xlsx', index=False)
test_data.to_excel('test_data.xlsx', index=False)

print(f"\n训练集大小: {len(X_train)} (90%)")
print(f"测试集大小: {len(X_test)} (10%)")

print("\n训练集标签分布:")
print(y_train.value_counts())
print("\n测试集标签分布:")
print(y_test.value_counts())

print("\n✓ 数据集已成功划分并保存!")
print("  - train_data.xlsx")
print("  - test_data.xlsx")
