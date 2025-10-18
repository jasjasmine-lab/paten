#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用平衡数据集训练 RoBERTa + BiLSTM + Cross-Attention 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import json
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 从原始模型脚本导入必要的类
from roberta_bilstm_crossattention_model import (
    PatentDatasetWithFeatures,
    CrossAttention,
    RoBERTaBiLSTMCrossAttention,
    train_epoch,
    evaluate
)

def main():
    """主函数 - 使用平衡数据集训练"""
    print("=" * 60)
    print("RoBERTa + BiLSTM + Cross-Attention 专利质量分级模型")
    print("使用平衡数据集训练")
    print("=" * 60)
    
    # 超参数设置
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 15  # 增加训练轮数
    MAX_LENGTH = 256
    
    # 加载分词器
    print("\n1. 加载RoBERTa分词器...")
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    
    # 创建数据集（使用平衡后的训练数据）
    print("\n2. 加载平衡后的数据集...")
    print("   训练集: train_data_balanced.xlsx (已平衡)")
    print("   测试集: test_data_clean.xlsx (保持原始分布)")
    
    train_dataset = PatentDatasetWithFeatures('train_data_balanced.xlsx', tokenizer, max_length=MAX_LENGTH)
    test_dataset = PatentDatasetWithFeatures('test_data_clean.xlsx', tokenizer, max_length=MAX_LENGTH, is_test=True)
    
    # 获取数值特征维度
    numerical_dim = train_dataset.numerical_features.shape[1]
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    print("\n3. 初始化模型...")
    model = RoBERTaBiLSTMCrossAttention(
        num_classes=3,
        lstm_hidden_size=256,
        lstm_layers=2,
        numerical_dim=numerical_dim,
        fusion_hidden_dim=256,
        dropout_rate=0.3
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # 训练模型
    print("\n4. 开始训练（使用平衡数据）...")
    print("-" * 60)
    
    best_accuracy = 0
    best_f1_macro = 0
    training_history = []
    
    for epoch in range(EPOCHS):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
        
        # 计算F1分数
        from sklearn.metrics import f1_score
        f1_macro = f1_score(labels, predictions, average='macro')
        
        # 调整学习率
        scheduler.step(test_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"  训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"  测试 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1-Macro: {f1_macro:.4f}")
        
        # 保存训练历史
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'f1_macro': f1_macro
        })
        
        # 保存最佳模型（基于F1-macro分数）
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_balanced_model.pth')
            print(f"  ✓ 保存最佳模型 (F1-Macro: {best_f1_macro:.4f})")
        
        print("-" * 60)
    
    # 加载最佳模型进行最终评估
    print("\n5. 最终评估（使用最佳模型）...")
    model.load_state_dict(torch.load('best_balanced_model.pth'))
    test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
    
    # 生成详细的分类报告
    label_names = ['A', 'B', 'C']
    print("\n分类报告:")
    report = classification_report(labels, predictions, target_names=label_names)
    print(report)
    
    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    print("\n混淆矩阵:")
    print("预测→  A    B    C")
    for i, label in enumerate(label_names):
        print(f"{label}    {cm[i]}")
    
    # 计算各类别的详细指标
    report_dict = classification_report(labels, predictions, target_names=label_names, output_dict=True)
    
    print("\n" + "=" * 60)
    print("训练完成（使用平衡数据）!")
    print(f"最佳测试准确率: {best_accuracy:.4f}")
    print(f"最佳F1-Macro分数: {best_f1_macro:.4f}")
    print("=" * 60)
    
    # 比较平衡前后的改进
    print("\n改进对比:")
    print("-" * 40)
    print("原始模型（不平衡数据）:")
    print("  准确率: 71.08%")
    print("  B类召回率: 0%")
    print("  C类召回率: 11%")
    print("-" * 40)
    print("平衡后模型:")
    print(f"  准确率: {test_acc:.2%}")
    print(f"  B类召回率: {report_dict['B']['recall']:.2%}")
    print(f"  C类召回率: {report_dict['C']['recall']:.2%}")
    print("-" * 40)
    
    # 保存结果
    results = {
        'best_accuracy': float(best_accuracy),
        'best_f1_macro': float(best_f1_macro),
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'training_history': training_history,
        'data_info': {
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'balanced': True,
            'samples_per_class': 1084
        }
    }
    
    with open('balanced_model_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n模型和结果已保存:")
    print("  - best_balanced_model.pth (平衡数据训练的模型权重)")
    print("  - balanced_model_results.json (评估结果)")
    
    # 生成改进总结
    print("\n" + "=" * 60)
    print("关键改进点:")
    print("1. 使用欠采样平衡了训练数据（每类1084样本）")
    print("2. 训练集从9114减少到3252样本，但类别分布均衡")
    print("3. 预期B类和C类的识别能力显著提升")
    print("4. F1-Macro分数作为主要评估指标（对各类别公平）")
    print("=" * 60)

if __name__ == "__main__":
    main()
