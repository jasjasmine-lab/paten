#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分块RoBERTa + BiLSTM + Cross-Attention + ResNet 模型 V3
改进：
1. 增加隐藏层维度到384
2. 调整块大小到768，重叠256
3. 集成ResNet增强文本和融合特征
"""

# 设置离线模式，使用本地缓存
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

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
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 从原模型导入数据集类和Cross-Attention
from roberta_bilstm_crossattention_model_v2 import PatentDatasetV2, CrossAttention

# =============================================================================
# ResNet模块定义
# =============================================================================

class ResidualBlock(nn.Module):
    """残差块，用于特征增强"""
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # 残差连接
        residual = x
        
        # 第一层
        out = self.layer_norm1(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        
        # 第二层
        out = self.fc2(out)
        out = self.dropout2(out)
        
        # 残差连接
        out = out + residual
        out = self.layer_norm2(out)
        
        return out

class TextResNet(nn.Module):
    """用于文本特征的ResNet模块（方案1）"""
    def __init__(self, hidden_dim, num_blocks=3, dropout_rate=0.1):
        super(TextResNet, self).__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) 
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class FusionResNet(nn.Module):
    """融合后的ResNet增强模块（方案2）"""
    def __init__(self, fusion_dim, num_blocks=2, dropout_rate=0.2):
        super(FusionResNet, self).__init__()
        
        # 投影层
        self.input_proj = nn.Linear(fusion_dim, fusion_dim)
        
        # ResNet块
        self.res_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.res_blocks.append(ResidualBlock(fusion_dim, dropout_rate))
        
        # 输出投影
        self.output_proj = nn.Linear(fusion_dim, fusion_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.res_blocks:
            x = block(x)
            
        x = self.output_proj(x)
        return x

# =============================================================================
# 改进的Cross-Attention模块
# =============================================================================

class ImprovedCrossAttention(nn.Module):
    """增强的Cross Attention机制，用于融合文本和数值特征"""
    
    def __init__(self, text_dim, num_dim, hidden_dim):
        super(ImprovedCrossAttention, self).__init__()
        
        # 投影层
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.num_proj = nn.Linear(num_dim, hidden_dim)
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, text_features, num_features):
        """
        Args:
            text_features: [batch_size, seq_len, text_dim]
            num_features: [batch_size, num_dim]
        Returns:
            fused_features: [batch_size, hidden_dim]
        """
        batch_size = text_features.size(0)
        
        # 投影到同一维度
        text_proj = self.text_proj(text_features)  # [batch_size, seq_len, hidden_dim]
        num_proj = self.num_proj(num_features).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 多头注意力
        attended_features, attention_weights = self.multihead_attn(
            num_proj, text_proj, text_proj
        )
        
        # 残差连接和层归一化
        attended_features = self.layer_norm1(attended_features + num_proj)
        
        # 前馈网络
        ffn_output = self.ffn(attended_features)
        fused_features = self.layer_norm2(attended_features + self.dropout(ffn_output))
        
        # 压缩维度
        fused_features = fused_features.squeeze(1)
        
        return fused_features, attention_weights

# =============================================================================
# 主模型
# =============================================================================

class ChunkedRoBERTaBiLSTMCrossAttentionWithResNet(nn.Module):
    """分块处理的RoBERTa + BiLSTM + Cross-Attention + ResNet模型"""
    
    def __init__(self, num_classes=3, lstm_hidden_size=384, lstm_layers=2, 
                 numerical_dim=7, fusion_hidden_dim=384, dropout_rate=0.2,
                 aggregation='attention', use_resnet=True, 
                 resnet_text_blocks=3, resnet_fusion_blocks=2):
        super(ChunkedRoBERTaBiLSTMCrossAttentionWithResNet, self).__init__()
        
        # 加载RoBERTa
        self.roberta = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.roberta_hidden_size = self.roberta.config.hidden_size
        
        # 启用梯度检查点以节省显存
        self.roberta.gradient_checkpointing_enable()
        
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
        self.use_resnet = use_resnet
        
        # ResNet模块
        if use_resnet:
            # 文本特征ResNet（方案1）
            self.text_resnet = TextResNet(
                hidden_dim=lstm_hidden_size * 2,
                num_blocks=resnet_text_blocks,
                dropout_rate=dropout_rate
            )
        
        # 块聚合层
        if aggregation == 'attention':
            self.chunk_attention = nn.MultiheadAttention(
                lstm_hidden_size * 2,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.global_query = nn.Parameter(torch.randn(1, 1, lstm_hidden_size * 2))
            nn.init.xavier_uniform_(self.global_query)
        
        # 改进的Cross Attention
        self.cross_attention = ImprovedCrossAttention(
            text_dim=lstm_hidden_size * 2,
            num_dim=numerical_dim,
            hidden_dim=fusion_hidden_dim
        )
        
        # 融合后ResNet（方案2）
        if use_resnet:
            self.fusion_resnet = FusionResNet(
                fusion_dim=fusion_hidden_dim,
                num_blocks=resnet_fusion_blocks,
                dropout_rate=dropout_rate
            )
        
        # 分类器
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
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
                with autocast(enabled=False):  # RoBERTa可能不支持混合精度
                    roberta_outputs = self.roberta(
                        input_ids=chunk_input,
                        attention_mask=chunk_mask
                    )
                sequence_output = roberta_outputs.last_hidden_state
                
                # BiLSTM处理
                lstm_output, _ = self.bilstm(sequence_output)
                
                # 取[CLS] token输出
                cls_embedding = lstm_output[:, 0, :]
                
                # 应用文本ResNet（如果启用）
                if self.use_resnet:
                    cls_embedding = self.text_resnet(cls_embedding)
                
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
        
        # 应用融合ResNet（如果启用）
        if self.use_resnet:
            fused_features = self.fusion_resnet(fused_features)
        
        # 分类
        fused_features = self.dropout(fused_features)
        logits = self.classifier(fused_features)
        
        return logits

# =============================================================================
# 训练函数
# =============================================================================

def train_epoch_with_amp(model, dataloader, criterion, optimizer, scaler, device, accumulation_steps=8):
    """使用混合精度和梯度累积的训练"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    progress_bar = tqdm(dataloader, desc="Training")
    
    for i, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        numerical_features = batch['numerical_features'].to(device)
        labels = batch['label'].to(device)
        
        # 混合精度前向传播
        with autocast():
            outputs = model(input_ids, attention_mask, numerical_features)
            loss = criterion(outputs, labels) / accumulation_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # 统计
        total_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
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

# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    print("=" * 80)
    print("分块RoBERTa + BiLSTM + Cross-Attention + ResNet 专利质量分级模型 V3")
    print("改进：增大隐藏层 + 更大块 + ResNet增强")
    print("=" * 80)
    
    # 超参数
    BATCH_SIZE = 8  # 由于模型更大，减小批次
    LEARNING_RATE = 3e-5  # 更保守的学习率
    EPOCHS = 150
    CHUNK_SIZE = 512  # RoBERTa最大支持512 tokens
    OVERLAP = 128  # 块之间的重叠
    MAX_CHUNKS = 16  # 增加块数以补偿较小的块大小
    AGGREGATION = 'attention'
    ACCUMULATION_STEPS = 4  # 梯度累积
    
    # 模型架构参数
    lstm_hidden_size = 384  # 增加到384
    lstm_layers = 2
    fusion_hidden_dim = 384  # 增加到384
    dropout_rate = 0.1
    resnet_text_blocks = 3
    resnet_fusion_blocks = 2
    
    print("\n超参数设置:")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  有效批次大小: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  块大小: {CHUNK_SIZE} tokens")
    print(f"  重叠: {OVERLAP} tokens")
    print(f"  最大块数: {MAX_CHUNKS}")
    print(f"  聚合策略: {AGGREGATION}")
    print(f"  LSTM隐藏层: {lstm_hidden_size}")
    print(f"  融合隐藏层: {fusion_hidden_dim}")
    print(f"  ResNet文本块: {resnet_text_blocks}")
    print(f"  ResNet融合块: {resnet_fusion_blocks}")
    
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
    model = ChunkedRoBERTaBiLSTMCrossAttentionWithResNet(
        num_classes=3,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        numerical_dim=7,
        fusion_hidden_dim=fusion_hidden_dim,
        dropout_rate=dropout_rate,
        aggregation=AGGREGATION,
        use_resnet=True,
        resnet_text_blocks=resnet_text_blocks,
        resnet_fusion_blocks=resnet_fusion_blocks
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
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=0.01,
        eps=1e-6
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader) // ACCUMULATION_STEPS,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # 混合精度训练
    scaler = GradScaler()
    
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
        train_loss, train_acc = train_epoch_with_amp(
            model, train_loader, criterion, optimizer, scaler, device, 
            accumulation_steps=ACCUMULATION_STEPS
        )
        
        # 学习率调度
        scheduler.step()
        
        # 验证
        test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
        
        # 记录
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['test_loss'].append(test_loss)
        training_history['test_acc'].append(test_acc)
        
        print(f"\n训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"测试 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_accuracy': best_accuracy,
            }, 'best_model_v3_resnet.pth')
            print(f"✓ 保存最佳模型 (Accuracy: {best_accuracy:.4f})")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'training_history': training_history,
            }, f'checkpoint_v3_epoch_{epoch+1}.pth')
        
        print("-" * 80)
    
    # 最终评估
    print("\n5. 最终评估...")
    checkpoint = torch.load('best_model_v3_resnet.pth')
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
            'accumulation_steps': ACCUMULATION_STEPS,
            'effective_batch_size': BATCH_SIZE * ACCUMULATION_STEPS,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'chunk_size': CHUNK_SIZE,
            'overlap': OVERLAP,
            'max_chunks': MAX_CHUNKS,
            'aggregation': AGGREGATION,
            'lstm_hidden_size': lstm_hidden_size,
            'lstm_layers': lstm_layers,
            'fusion_hidden_dim': fusion_hidden_dim,
            'dropout_rate': dropout_rate,
            'resnet_text_blocks': resnet_text_blocks,
            'resnet_fusion_blocks': resnet_fusion_blocks
        }
    }
    
    with open('model_v3_resnet_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n模型和结果已保存:")
    print("  - best_model_v3_resnet.pth (最佳模型权重)")
    print("  - checkpoint_v3_epoch_*.pth (检查点)")
    print("  - model_v3_resnet_results.json (评估结果)")

if __name__ == "__main__":
    main()
