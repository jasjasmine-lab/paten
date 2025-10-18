#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用传统机器学习方法的专利质量分级模型
不依赖于外部预训练模型，使用TF-IDF + 随机森林
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def prepare_text_features(df):
    """准备文本特征"""
    # 选择重要的文本列
    text_columns = ['标题 (中文)', '摘要 (中文)', '首项权利要求', 
                   '技术功效句', '用途']
    
    # 合并文本
    texts = []
    for idx, row in df.iterrows():
        combined_text = ""
        for col in text_columns:
            if col in df.columns and pd.notna(row[col]):
                combined_text += str(row[col]) + " "
        
        if not combined_text.strip():
            combined_text = "专利文本"
        
        texts.append(combined_text)
    
    return texts

def extract_numerical_features(df):
    """提取数值特征"""
    numerical_features = []
    
    # 选择可能有用的数值列
    num_columns = ['权利要求数量', '首权字数', '申请人数量', '发明(设计)人数量',
                  '简单同族个数', '扩展同族个数', 'DocDB同族个数']
    
    for col in num_columns:
        if col in df.columns:
            # 填充缺失值
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
            numerical_features.append(df[col].values.reshape(-1, 1))
    
    if numerical_features:
