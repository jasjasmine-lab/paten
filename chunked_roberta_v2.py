#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分块RoBERTa + BiLSTM + Cross-Attention 模型
支持长文本通过分块处理后聚合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import json
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 从原模型导入数据集类
from roberta_bilstm_crossattention_model_v2 import PatentDatasetV2, CrossAttention

class ChunkedRoBERTaBiLSTMCrossAttention(nn.Module):
    """分块处理的RoBERTa + BiLSTM + Cross-Attention模型"""
    
    def __init__(self, num_classes=3, lstm_hidden_size=256, lstm_layers=2, 
                 numerical_dim=7, fusion_hidden_dim=256, dropout_rate=0.5,
                 aggregation='attention'):
        super(ChunkedRoBERTaBiLSTMCrossAttention, self).__init__()
        
        # 加载RoBERTa
        self.roberta = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.roberta_hidden_size = self.roberta.config.hidden_size
        
        # 冻结embedding层
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        
        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.roberta_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        self.aggregation = aggregation
        self.lstm_hidden_size = lstm_hidden_size
        
        # 块聚合层
        if aggregation == 'attention':
            self.chunk_attention = nn.MultiheadAttention(
                lstm_hidden_size * 2,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.global_query = nn.Parameter(torch.randn(1, 1, lstm_hidden_size * 2))
        
        # Cross Attention
        self.cross_attention = CrossAttention(
            text_dim=lstm_hidden_size * 2,
            num_dim=numerical_dim,
            hidden_dim=fusion_hidden_dim
        )
        
        # 分类器
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, numerical_features):
        """
        Args:
            input_ids: [batch_size, max_chunks, chunk_size]
            attention_mask: [batch_size, max_chunks, chunk_size]
            numerical_features: [batch_size, numerical_dim]
        """
        batch_size, max_chunks, chunk_size = input_ids.shape
        
        # 处理每个块
        chunk_embeddings = []
        
        for i in range(max_chunks):
            chunk_input = input_ids[:, i, :]
            chunk_mask = attention_mask[:, i, :]
            
            # 检查有效块
            is_valid_chunk = chunk_mask.sum(dim=1) > 0
            
            if is_valid_chunk.any():
                # RoBERTa编码
                roberta_outputs = self.roberta(
                    input_ids=chunk_input,
                    attention_mask=chunk_mask
                )
                sequence_output = roberta_outputs.last_hidden_state
                
                # BiLSTM处理
                lstm_output, _ = self.bilstm(sequence_output)
                
                # 取[CLS] token输出
                cls_embedding = lstm_output[:, 0, :]
                
                # 对无效块置零
                cls_embedding = cls_embedding * is_valid_chunk.unsqueeze(1).float()
            else:
                cls_embedding = torch.zeros(batch_size, self.lstm_hidden_size * 2).to(device)
            
            chunk_embeddings.append(cls_embedding)
        
        # Stack块embeddings
        chunk_embeddings = torch.stack(chunk_embeddings, dim=1)
        
        # 聚合块
        if self.aggregation == 'max':
            text_features, _ = torch.max(chunk_embeddings, dim=1)
        
        elif self.aggregation == 'mean':
            valid_mask = (chunk_embeddings.sum(dim=2) != 0).float()
            valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
            weighted_sum = (chunk_embeddings * valid_mask.unsqueeze(2)).sum(dim=1)
            text_features = weighted_sum / valid_counts
        
        elif self.aggregation == 'attention':
            query = self.global_query.expand(batch_size, 1, self.lstm_hidden_size * 2)
            valid_mask = (chunk_embeddings.sum(dim=2) != 0)
            attn_mask = ~valid_mask
            
            text_features, _ = self.chunk_attention(
                query, 
                chunk_embeddings, 
                chunk_embeddings,
                key_padding_mask=attn_mask
            )
            text_features = text_features.squeeze(1)
        
        # Cross Attention融合
        text_features_for_ca = text_features.unsqueeze(1)
        fused_features, _ = self.cross_attention(text_features_for_ca, numerical_features)
        
        # 分类
        fused_features = self.dropout(fused_features)
        logits = self.classifier(fused_features)
        
        return logits

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        numerical_features = batch['numerical_features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, numerical_features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, numerical_features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            current_acc = accuracy_score(all_labels, all_predictions)
            progress_bar.set_postfix({'acc': f'{current_acc:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, all_predictions, all_labels

def main():
    """主函数"""
    print("=" * 80)
    print("分块RoBERTa + BiLSTM + Cross-Attention 专利质量分级模型")
    print("支持长文本通过分块处理后聚合")
    print("=" * 80)
    
    # 超参数
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    EPOCHS = 50
    CHUNK_SIZE = 512
    OVERLAP = 128
    MAX_CHUNKS = 16
    AGGREGATION = 'attention'
    
    print("\n超参数设置:")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  块大小: {CHUNK_SIZE} tokens")
    print(f"  重叠: {OVERLAP} tokens")
    print(f"  最大块数: {MAX_CHUNKS}")
    print(f"  聚合策略: {AGGREGATION}")
    
    # 加载分词器
    print("\n1. 加载RoBERTa分词器...")
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    
    # 创建数据集
    print("\n2. 加载数据集...")
    train_dataset = PatentDatasetV2(
        'train_data_prepared.xlsx', 
        tokenizer, 
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        max_chunks=MAX_CHUNKS
    )
    
    test_dataset = PatentDatasetV2(
        'test_data_prepared.xlsx', 
        tokenizer,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        max_chunks=MAX_CHUNKS,
        is_test=True
    )
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 初始化模型
    print("\n3. 初始化模型...")
    model = ChunkedRoBERTaBiLSTMCrossAttention(
        num_classes=3,
        lstm_hidden_size=256,
        lstm_layers=2,
        numerical_dim=7,
        fusion_hidden_dim=256,
        dropout_rate=0.3,
        aggregation=AGGREGATION
    ).to(device)
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {total_params - trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 训练
    print("\n4. 开始训练...")
    print("-" * 80)
    
    best_accuracy = 0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
        
        # 学习率调度
        scheduler.step()
        
        # 记录
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['test_loss'].append(test_loss)
        training_history['test_acc'].append(test_acc)
        
        print(f"\n训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"测试 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        
        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
            }, 'best_chunked_model.pth')
            print(f"✓ 保存最佳模型 (Accuracy: {best_accuracy:.4f})")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_history': training_history,
            }, f'checkpoint_epoch_{epoch+1}.pth')
        
        print("-" * 80)
    
    # 最终评估
    print("\n5. 最终评估...")
    checkpoint = torch.load('best_chunked_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
    
    # 分类报告
    label_names = ['A', 'B', 'C']
    print("\n分类报告:")
    print(classification_report(labels, predictions, target_names=label_names))
    
    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    print("\n混淆矩阵:")
    print("预测→  A    B    C")
    for i, label in enumerate(label_names):
        print(f"{label}    {cm[i]}")
    
    print("\n" + "=" * 80)
    print(f"训练完成! 最佳测试准确率: {best_accuracy:.4f}")
    print("=" * 80)
    
    # 保存结果
    results = {
        'best_accuracy': float(best_accuracy),
        'classification_report': classification_report(labels, predictions, 
                                                    target_names=label_names, 
                                                    output_dict=True),
        'confusion_matrix': cm.tolist(),
        'training_history': training_history,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'chunk_size': CHUNK_SIZE,
            'overlap': OVERLAP,
            'max_chunks': MAX_CHUNKS,
            'aggregation': AGGREGATION
        }
    }
    
    with open('chunked_model_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n模型和结果已保存:")
    print("  - best_chunked_model.pth (最佳模型权重)")
    print("  - checkpoint_epoch_*.pth (检查点)")
    print("  - chunked_model_results.json (评估结果)")

if __name__ == "__main__":
    main()
