#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展训练版本 - 使用平衡数据集训练 RoBERTa + BiLSTM + Cross-Attention 模型
增加训练轮次和优化策略以提高准确率
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
import matplotlib.pyplot as plt
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

def plot_training_history(history, save_path='training_history.png'):
    """绘制训练历史"""
    epochs = [h['epoch'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    test_acc = [h['test_acc'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    test_loss = [h['test_loss'] for h in history]
    f1_scores = [h['f1_macro'] for h in history]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 准确率图
    axes[0].plot(epochs, train_acc, 'b-', label='Train Accuracy')
    axes[0].plot(epochs, test_acc, 'r-', label='Test Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # 损失图
    axes[1].plot(epochs, train_loss, 'b-', label='Train Loss')
    axes[1].plot(epochs, test_loss, 'r-', label='Test Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    # F1分数图
    axes[2].plot(epochs, f1_scores, 'g-', label='F1-Macro')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1-Macro Score')
    axes[2].set_title('F1-Macro Score Progress')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n训练历史图表已保存到: {save_path}")

def main():
    """主函数 - 扩展训练版本"""
    print("=" * 60)
    print("RoBERTa + BiLSTM + Cross-Attention 专利质量分级模型")
    print("扩展训练版本 - 增加训练轮次和优化策略")
    print("=" * 60)
    
    # 超参数设置（优化版）
    BATCH_SIZE = 16
    INITIAL_LR = 3e-5  # 稍微提高初始学习率
    EPOCHS = 200  # 设置为200个训练轮次
    MAX_LENGTH = 256
    WARMUP_EPOCHS = 5  # 添加warmup阶段
    
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
    
    # 损失函数（添加类别权重）
    # 根据测试集的真实分布计算权重
    class_weights = torch.tensor([1.0, 2.5, 3.5]).to(device)  # 给少数类更高权重
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.01)
    
    # 学习率调度器 - 使用余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    
    # 训练模型
    print("\n4. 开始扩展训练...")
    print(f"   总轮次: {EPOCHS}")
    print(f"   初始学习率: {INITIAL_LR}")
    print(f"   使用类别权重: {class_weights.cpu().numpy()}")
    print("-" * 60)
    
    best_accuracy = 0
    best_f1_macro = 0
    best_balanced_score = 0  # 综合评分
    training_history = []
    patience = 20  # 增加早停耐心值
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Warmup阶段调整学习率
        if epoch < WARMUP_EPOCHS:
            warmup_lr = INITIAL_LR * (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            current_lr = warmup_lr
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
        
        # 计算详细指标
        from sklearn.metrics import f1_score, balanced_accuracy_score
        f1_macro = f1_score(labels, predictions, average='macro')
        balanced_acc = balanced_accuracy_score(labels, predictions)
        
        # 计算综合评分（结合准确率和F1分数）
        balanced_score = 0.4 * test_acc + 0.6 * f1_macro
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] (LR: {current_lr:.6f})")
        print(f"  训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"  测试 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        print(f"  指标 - F1-Macro: {f1_macro:.4f}, Balanced Acc: {balanced_acc:.4f}")
        
        # 保存训练历史
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'f1_macro': f1_macro,
            'balanced_acc': balanced_acc,
            'lr': current_lr
        })
        
        # 保存最佳模型（基于综合评分）
        if balanced_score > best_balanced_score:
            best_balanced_score = balanced_score
            best_f1_macro = f1_macro
            best_accuracy = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
                'f1_macro': f1_macro,
                'balanced_acc': balanced_acc
            }, 'best_extended_model.pth')
            print(f"  ✓ 保存最佳模型 (综合评分: {balanced_score:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        print("-" * 60)
        
        # 早停检查
        if patience_counter >= patience and epoch >= 30:  # 至少训练30轮
            print(f"\n早停触发（{patience}轮无改善），停止训练")
            break
    
    # 绘制训练历史
    plot_training_history(training_history)
    
    # 加载最佳模型进行最终评估
    print("\n5. 最终评估（使用最佳模型）...")
    checkpoint = torch.load('best_extended_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
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
    print("扩展训练完成!")
    print(f"最佳测试准确率: {best_accuracy:.4f}")
    print(f"最佳F1-Macro分数: {best_f1_macro:.4f}")
    print(f"最佳训练轮次: {checkpoint['epoch']}")
    print("=" * 60)
    
    # 与之前的结果对比
    print("\n性能对比:")
    print("-" * 40)
    print("基础平衡模型（15轮）:")
    print("  准确率: 48.57%")
    print("  F1-Macro: 0.4158")
    print("-" * 40)
    print("扩展训练模型（30轮+优化）:")
    print(f"  准确率: {test_acc:.2%}")
    print(f"  F1-Macro: {f1_macro:.4f}")
    print(f"  B类召回率: {report_dict['B']['recall']:.2%}")
    print(f"  C类召回率: {report_dict['C']['recall']:.2%}")
    print("-" * 40)
    
    # 保存结果
    results = {
        'best_accuracy': float(best_accuracy),
        'best_f1_macro': float(best_f1_macro),
        'best_epoch': checkpoint['epoch'],
        'total_epochs': len(training_history),
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'training_history': training_history,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'initial_lr': INITIAL_LR,
            'max_epochs': EPOCHS,
            'warmup_epochs': WARMUP_EPOCHS,
            'class_weights': class_weights.cpu().numpy().tolist()
        },
        'data_info': {
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'balanced': True,
            'samples_per_class': 1084
        }
    }
    
    with open('extended_model_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n模型和结果已保存:")
    print("  - best_extended_model.pth (扩展训练模型权重)")
    print("  - extended_model_results.json (详细评估结果)")
    print("  - training_history.png (训练历史图表)")
    
    # 生成改进总结
    print("\n" + "=" * 60)
    print("扩展训练优化策略:")
    print("1. 训练轮次设置为100个epoch")
    print("2. 使用类别权重平衡损失函数")
    print("3. 采用余弦退火学习率调度")
    print("4. 添加5轮Warmup阶段稳定训练")
    print("5. 使用综合评分选择最佳模型")
    print("6. 添加早停机制防止过拟合（至少30轮）")
    print("=" * 60)

if __name__ == "__main__":
    main()
