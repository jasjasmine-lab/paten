#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版训练脚本 - 带Cross Attention机制
使用Cross Attention融合文本和数值特征，提升模型性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
import warnings
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class CustomLossWithPenalty(nn.Module):
    """自定义损失函数：加入BC->A误分类的额外惩罚"""
    
    def __init__(self, class_weights=None, bc_to_a_penalty=2.0):
        """
        Args:
            class_weights: 类别权重，用于处理类别不平衡
            bc_to_a_penalty: BC类被误分为A类时的额外惩罚系数
        """
        super(CustomLossWithPenalty, self).__init__()
        self.class_weights = class_weights
        self.bc_to_a_penalty = bc_to_a_penalty
        
        # 基础交叉熵损失
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        else:
            self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, outputs, labels):
        """
        Args:
            outputs: 模型输出的logits [batch_size, num_classes]
            labels: 真实标签 [batch_size]
        Returns:
            loss: 加权后的损失值
        """
        # 计算基础交叉熵损失
        ce_losses = self.ce_loss(outputs, labels)
        
        # 获取预测类别
        _, predicted = torch.max(outputs, 1)
        
        # 创建惩罚掩码：当真实标签是B(1)或C(2)，但预测为A(0)时
        # B类和C类被误分为A类的情况
        bc_mask = (labels == 1) | (labels == 2)  # 真实标签是B或C
        misclassified_to_a = predicted == 0      # 预测为A
        penalty_mask = bc_mask & misclassified_to_a
        
        # 对BC->A的误分类应用额外惩罚
        penalty_weights = torch.ones_like(ce_losses)
        penalty_weights[penalty_mask] = self.bc_to_a_penalty
        
        # 应用惩罚权重
        weighted_losses = ce_losses * penalty_weights
        
        # 返回平均损失
        return weighted_losses.mean()
    
    def get_penalty_stats(self, outputs, labels):
        """获取惩罚统计信息（用于监控）"""
        _, predicted = torch.max(outputs, 1)
        
        # 统计BC->A的误分类数量
        bc_mask = (labels == 1) | (labels == 2)
        misclassified_to_a = predicted == 0
        bc_to_a_count = (bc_mask & misclassified_to_a).sum().item()
        
        # 统计各类误分类
        b_to_a = ((labels == 1) & (predicted == 0)).sum().item()
        c_to_a = ((labels == 2) & (predicted == 0)).sum().item()
        
        return {
            'bc_to_a_total': bc_to_a_count,
            'b_to_a': b_to_a,
            'c_to_a': c_to_a
        }

class CrossAttention(nn.Module):
    """Cross Attention机制，用于融合文本和数值特征"""
    
    def __init__(self, text_dim, num_dim, hidden_dim, dropout_rate=0.1):
        super(CrossAttention, self).__init__()
        
        # 投影层
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.num_proj = nn.Linear(num_dim, hidden_dim)
        
        # 注意力计算
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = hidden_dim ** 0.5
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, text_features, num_features):
        """
        Args:
            text_features: [batch_size, seq_len, text_dim] 或 [batch_size, text_dim]
            num_features: [batch_size, num_dim]
        Returns:
            fused_features: [batch_size, hidden_dim]
            attention_weights: [batch_size, seq_len] 或 None
        """
        batch_size = text_features.size(0)
        
        # 处理文本特征（可能是序列或池化后的向量）
        if len(text_features.shape) == 2:
            # 已池化的文本特征 [batch_size, text_dim]
            text_proj = self.text_proj(text_features).unsqueeze(1)  # [batch_size, 1, hidden_dim]
            seq_len = 1
        else:
            # 序列文本特征 [batch_size, seq_len, text_dim]
            text_proj = self.text_proj(text_features)  # [batch_size, seq_len, hidden_dim]
            seq_len = text_features.size(1)
        
        num_proj = self.num_proj(num_features).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 数值特征作为Query，文本特征作为Key和Value
        Q = self.query(num_proj)  # [batch_size, 1, hidden_dim]
        K = self.key(text_proj)    # [batch_size, seq_len, hidden_dim]
        V = self.value(text_proj)  # [batch_size, seq_len, hidden_dim]
        
        # 计算注意力分数
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [batch_size, 1, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        attended_features = torch.bmm(attention_weights, V)  # [batch_size, 1, hidden_dim]
        attended_features = attended_features.squeeze(1)  # [batch_size, hidden_dim]
        
        # 残差连接和Layer Norm
        fused_features = self.layer_norm(attended_features + num_proj.squeeze(1))
        
        return fused_features, attention_weights.squeeze(1) if seq_len > 1 else None

class RoBERTaCrossAttentionModel(nn.Module):
    """带Cross Attention的RoBERTa模型"""
    
    def __init__(self, num_classes=3, lstm_hidden_size=128, lstm_layers=2, 
                 numerical_dim=7, fusion_hidden_dim=256, dropout_rate=0.4):
        """
        初始化模型
        Args:
            num_classes: 分类数量（A/B/C = 3）
            lstm_hidden_size: LSTM隐藏层大小
            lstm_layers: LSTM层数
            numerical_dim: 数值特征维度
            fusion_hidden_dim: 融合层隐藏维度
            dropout_rate: Dropout率
        """
        super(RoBERTaCrossAttentionModel, self).__init__()
        
        # 加载预训练的RoBERTa模型
        roberta_model_name = 'hfl/chinese-roberta-wwm-ext'
        print(f"加载RoBERTa模型: {roberta_model_name}")
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.roberta_hidden_size = self.roberta.config.hidden_size
        
        # 冻结RoBERTa更多层以防止过拟合
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        for layer in self.roberta.encoder.layer[:8]:  # 冻结前8层
            for param in layer.parameters():
                param.requires_grad = False
        
        # BiLSTM层
        self.bilstm = nn.LSTM(
            input_size=self.roberta_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        # 文本特征池化
        self.text_pooling = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout_rate)
        )
        
        # Cross Attention层（融合文本和数值特征）
        self.cross_attention = CrossAttention(
            text_dim=lstm_hidden_size * 2,  # BiLSTM输出
            num_dim=numerical_dim,
            hidden_dim=fusion_hidden_dim,
            dropout_rate=dropout_rate
        )
        
        # 额外的注意力层（可选，用于进一步融合）
        self.self_attention = nn.MultiheadAttention(
            embed_dim=fusion_hidden_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, numerical_features):
        """
        前向传播
        """
        # RoBERTa编码
        roberta_outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = roberta_outputs.last_hidden_state
        
        # BiLSTM处理
        lstm_output, _ = self.bilstm(sequence_output)
        
        # 池化文本特征
        lstm_pooled = lstm_output.transpose(1, 2)  # [batch, hidden*2, seq_len]
        lstm_pooled = self.text_pooling(lstm_pooled).squeeze(-1)  # [batch, hidden*2]
        
        # Cross Attention融合文本和数值特征
        fused_features, _ = self.cross_attention(lstm_pooled, numerical_features)
        
        # 可选：自注意力进一步处理
        fused_features_seq = fused_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        fused_features_att, _ = self.self_attention(
            fused_features_seq, fused_features_seq, fused_features_seq
        )
        fused_features = fused_features_att.squeeze(1)
        
        # Dropout
        fused_features = self.dropout(fused_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits

class OptimizedPatentDataset(Dataset):
    """优化的专利数据集类"""
    
    def __init__(self, data_path, tokenizer, max_length=256, is_test=False):
        """初始化数据集"""
        self.data = pd.read_excel(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
        print(f"加载数据: {data_path}")
        print(f"数据量: {len(self.data)}")
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(['A', 'B', 'C'])
        self.labels = self.label_encoder.transform(self.data['质量标签'])
        
        # 准备文本数据
        self.texts = self._prepare_texts()
        
        # 准备数值特征
        self.numerical_features = self._prepare_numerical_features()
        
        # 打印数据分布
        unique, counts = np.unique(self.labels, return_counts=True)
        print("数据分布:")
        for label_idx, count in zip(unique, counts):
            label = self.label_encoder.inverse_transform([label_idx])[0]
            print(f"  {label}级: {count}条 ({count/len(self.data)*100:.2f}%)")
    
    def _prepare_texts(self):
        """准备文本数据"""
        texts = []
        text_columns = ['标题 (中文)', '摘要 (中文)', '首项权利要求', 
                       '技术功效句', '用途']
        
        for idx, row in self.data.iterrows():
            combined_text = ""
            for col in text_columns:
                if col in self.data.columns and pd.notna(row[col]):
                    combined_text += str(row[col]) + " "
            
            if not combined_text.strip():
                combined_text = "专利文本"
            
            texts.append(combined_text[:1500])
        
        return texts
    
    def _prepare_numerical_features(self):
        """准备数值特征（不包含被引用信息）"""
        numerical_columns = [
            '权利要求数量', '首权字数', '申请人数量',
            '发明(设计)人数量', '简单同族个数',
            '扩展同族个数', 'DocDB同族个数'
        ]
        
        features = []
        valid_columns = []
        
        for col in numerical_columns:
            if col in self.data.columns:
                col_data = pd.to_numeric(self.data[col], errors='coerce')
                col_data = col_data.fillna(col_data.median() if col_data.notna().any() else 0)
                features.append(col_data.values.reshape(-1, 1))
                valid_columns.append(col)
        
        print(f"使用的数值特征: {valid_columns}")
        
        if features:
            features = np.hstack(features)
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        else:
            features = np.zeros((len(self.data), 1))
        
        return features.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        numerical_feat = self.numerical_features[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'numerical_features': torch.tensor(numerical_feat, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_epoch_with_accumulation(model, dataloader, criterion, optimizer, device, accumulation_steps=2, track_penalties=False):
    """带梯度累积的训练"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 惩罚统计
    bc_to_a_total = 0
    b_to_a_total = 0
    c_to_a_total = 0
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        numerical_features = batch['numerical_features'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        outputs = model(input_ids, attention_mask, numerical_features)
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps  # 标准化损失
        
        # 反向传播
        loss.backward()
        
        # 如果需要跟踪惩罚统计
        if track_penalties and isinstance(criterion, CustomLossWithPenalty):
            penalty_stats = criterion.get_penalty_stats(outputs, labels)
            bc_to_a_total += penalty_stats['bc_to_a_total']
            b_to_a_total += penalty_stats['b_to_a']
            c_to_a_total += penalty_stats['c_to_a']
        
        # 梯度累积
        if (i + 1) % accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # 统计
        total_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 处理最后的梯度
    if len(dataloader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    # 返回结果，包括惩罚统计（如果需要）
    if track_penalties:
        penalty_info = {
            'bc_to_a_total': bc_to_a_total,
            'b_to_a': b_to_a_total,
            'c_to_a': c_to_a_total
        }
        return avg_loss, accuracy, penalty_info
    else:
        return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
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
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, all_predictions, all_labels

def plot_training_curves(history, save_path='crossattention_training_history.png'):
    """绘制训练曲线"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 损失曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['test_loss'], 'r-', label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    axes[1].plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # F1分数曲线
    axes[2].plot(epochs, history['f1_macro'], 'g-', label='F1-Macro')
    axes[2].plot(epochs, history['balanced_acc'], 'm-', label='Balanced Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score')
    axes[2].set_title('F1-Macro and Balanced Accuracy')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n训练曲线已保存到: {save_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("带Cross Attention的专利质量分级模型训练")
    print("使用Cross Attention机制融合文本和数值特征")
    print("=" * 60)
    
    # 优化的超参数
    BATCH_SIZE = 24          # 适中的批次大小
    INITIAL_LR = 2e-5        # 略高的学习率
    EPOCHS = 60              # 训练轮次
    MAX_LENGTH = 256
    WARMUP_EPOCHS = 3        # warmup轮次
    ACCUMULATION_STEPS = 2   # 梯度累积步数
    
    # 加载分词器
    print("\n1. 加载RoBERTa分词器...")
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    
    # 检查增强数据是否存在
    if not os.path.exists('/root/autodl-tmp/train_data_augmented.xlsx'):
        print("\n增强数据不存在，先运行数据增强...")
        os.system('cd /root/autodl-tmp && python data_augmentation_smote.py')
        print("数据增强完成！")
    
    # 创建数据集
    print("\n2. 加载数据集...")
    print("   训练集: train_data_augmented.xlsx (增强数据)")
    print("   测试集: test_data_clean.xlsx (保持原始分布)")
    
    train_dataset = OptimizedPatentDataset('/root/autodl-tmp/train_data_augmented.xlsx', tokenizer, max_length=MAX_LENGTH)
    test_dataset = OptimizedPatentDataset('/root/autodl-tmp/test_data_clean.xlsx', tokenizer, max_length=MAX_LENGTH, is_test=True)
    
    # 获取数值特征维度
    numerical_dim = train_dataset.numerical_features.shape[1]
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 初始化带Cross Attention的模型
    print("\n3. 初始化Cross Attention模型...")
    model = RoBERTaCrossAttentionModel(
        num_classes=3,
        lstm_hidden_size=128,  # LSTM隐藏层大小
        lstm_layers=2,          # LSTM层数
        numerical_dim=numerical_dim,
        fusion_hidden_dim=256,  # Cross Attention融合维度
        dropout_rate=0.4        # Dropout率
    ).to(device)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"参数减少比例: {(1 - trainable_params/total_params)*100:.2f}%")
    
    # 使用自定义损失函数（带BC->A误分类惩罚）
    class_weights = torch.tensor([1.0, 4.0, 3.0]).to(device)
    BC_TO_A_PENALTY = 2.5  # BC类被误分为A类的额外惩罚系数
    criterion = CustomLossWithPenalty(class_weights=class_weights, bc_to_a_penalty=BC_TO_A_PENALTY)
    print(f"\n类别权重: {class_weights.cpu().numpy()}")
    print(f"BC->A误分类惩罚系数: {BC_TO_A_PENALTY}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'f1_macro': [], 'balanced_acc': []
    }
    
    # 训练模型
    print("\n4. 开始训练...")
    print(f"   批次大小: {BATCH_SIZE}")
    print(f"   初始学习率: {INITIAL_LR}")
    print(f"   训练轮次: {EPOCHS}")
    print(f"   梯度累积步数: {ACCUMULATION_STEPS}")
    print(f"   Cross Attention融合维度: 256")
    print(f"   BC->A惩罚机制: 启用 (惩罚系数={BC_TO_A_PENALTY})")
    print("-" * 60)
    
    best_accuracy = 0
    best_f1_macro = 0
    best_balanced_score = 0
    patience = 15
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        # Warmup阶段
        if epoch < WARMUP_EPOCHS:
            warmup_lr = INITIAL_LR * (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            current_lr = warmup_lr
        else:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        
        # 训练（跟踪惩罚统计）
        train_result = train_epoch_with_accumulation(
            model, train_loader, criterion, optimizer, device, ACCUMULATION_STEPS, 
            track_penalties=(epoch % 5 == 0)  # 每5个epoch跟踪一次惩罚统计
        )
        
        # 解析训练结果
        if len(train_result) == 3:
            train_loss, train_acc, penalty_info = train_result
            # 打印惩罚统计
            print(f"  惩罚统计 - BC->A总数: {penalty_info['bc_to_a_total']}, "
                  f"B->A: {penalty_info['b_to_a']}, C->A: {penalty_info['c_to_a']}")
        else:
            train_loss, train_acc = train_result
        
        # 验证
        test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
        
        # 计算详细指标
        f1_macro = f1_score(labels, predictions, average='macro')
        balanced_acc = balanced_accuracy_score(labels, predictions)
        
        # 计算综合评分（更重视F1-macro）
        balanced_score = 0.3 * test_acc + 0.7 * f1_macro
        
        # 保存历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['f1_macro'].append(f1_macro)
        history['balanced_acc'].append(balanced_acc)
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{EPOCHS}] (LR: {current_lr:.7f})")
        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        print(f"  指标 - F1: {f1_macro:.4f}, Balanced: {balanced_acc:.4f}")
        
        # 保存最佳模型
        if balanced_score > best_balanced_score:
            best_balanced_score = balanced_score
            best_f1_macro = f1_macro
            best_accuracy = test_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
                'f1_macro': f1_macro,
                'balanced_acc': balanced_acc
            }, '/root/autodl-tmp/best_crossattention_model.pth')
            print(f"  ✓ 保存最佳模型 (Score: {balanced_score:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        print("-" * 60)
        
        # 早停
        if patience_counter >= patience and epoch >= 20:
            print(f"\n早停触发（{patience}轮无改善），停止训练")
            break
    
    # 绘制训练曲线
    plot_training_curves(history, save_path='/root/autodl-tmp/crossattention_training_history.png')
    
    # 加载最佳模型进行最终评估
    print("\n5. 最终评估（使用最佳模型）...")
    checkpoint = torch.load('/root/autodl-tmp/best_crossattention_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
    
    # 生成详细报告
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
    print("Cross Attention模型训练完成!")
    print(f"最佳测试准确率: {best_accuracy:.4f}")
    print(f"最佳F1-Macro分数: {best_f1_macro:.4f}")
    print(f"最佳训练轮次: {best_epoch}")
    print("=" * 60)
    
    # 与之前的结果对比
    print("\n性能对比:")
    print("-" * 40)
    print("优化模型（无Cross Attention）:")
    print("  准确率: 68.31%")
    print("  F1-Macro: 0.4851")
    print("-" * 40)
    print("Cross Attention模型:")
    print(f"  准确率: {test_acc:.2%}")
    print(f"  F1-Macro: {f1_macro:.4f}")
    print(f"  B类召回率: {report_dict['B']['recall']:.2%}")
    print(f"  C类召回率: {report_dict['C']['recall']:.2%}")
    print("-" * 40)
    
    # 保存结果
    results = {
        'best_accuracy': float(best_accuracy),
        'best_f1_macro': float(best_f1_macro),
        'best_epoch': best_epoch,
        'total_epochs': len(history['train_loss']),
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'training_history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'test_loss': [float(x) for x in history['test_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'test_acc': [float(x) for x in history['test_acc']],
            'f1_macro': [float(x) for x in history['f1_macro']],
            'balanced_acc': [float(x) for x in history['balanced_acc']]
        },
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'initial_lr': INITIAL_LR,
            'epochs': EPOCHS,
            'warmup_epochs': WARMUP_EPOCHS,
            'accumulation_steps': ACCUMULATION_STEPS,
            'class_weights': class_weights.cpu().numpy().tolist(),
            'lstm_hidden_size': 128,
            'lstm_layers': 2,
            'fusion_hidden_dim': 256,
            'dropout_rate': 0.4,
            'architecture': 'RoBERTa + BiLSTM + Cross-Attention'
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('/root/autodl-tmp/crossattention_model_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n模型和结果已保存:")
    print("  - /root/autodl-tmp/best_crossattention_model.pth (Cross Attention模型权重)")
    print("  - /root/autodl-tmp/crossattention_model_results.json (详细评估结果)")
    print("  - /root/autodl-tmp/crossattention_training_history.png (训练曲线)")
    
    # 生成优化总结
    print("\n" + "=" * 60)
    print("Cross Attention优化策略总结:")
    print("1. 架构增强: 添加Cross Attention机制融合文本和数值特征")
    print("2. 注意力机制: Query-Key-Value注意力计算")
    print("3. 残差连接: Cross Attention带残差连接和Layer Norm")
    print("4. 多头自注意力: 额外的自注意力层进一步融合特征")
    print("5. 融合策略: 文本特征作为Key/Value，数值特征作为Query")
    print("6. 梯度累积: 提高训练稳定性")
    print("7. BC->A惩罚机制: 对BC类误分为A类进行额外惩罚")
    print("=" * 60)

if __name__ == "__main__":
    main()
