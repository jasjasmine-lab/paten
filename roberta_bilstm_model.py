#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoBERTa + BiLSTM + Attention 模型实现
用于专利质量分级（A/B/C）
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
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class PatentDataset(Dataset):
    """专利数据集类"""
    
    def __init__(self, data_path, tokenizer, max_length=512, is_test=False):
        """
        初始化数据集
        Args:
            data_path: Excel文件路径
            tokenizer: RoBERTa分词器
            max_length: 最大序列长度
            is_test: 是否为测试集
        """
        self.data = pd.read_excel(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        if not is_test:
            self.labels = self.label_encoder.fit_transform(self.data['质量标签'])
        else:
            # 使用训练集的编码器
            self.label_encoder.classes_ = np.array(['A', 'B', 'C'])
            self.labels = self.label_encoder.transform(self.data['质量标签'])
        
        # 准备文本数据（结合多个文本字段）
        self.texts = self._prepare_texts()
        
        print(f"数据集大小: {len(self.data)}")
        print(f"标签类别: {self.label_encoder.classes_}")
    
    def _prepare_texts(self):
        """准备文本数据，组合多个相关字段"""
        texts = []
        
        # 选择最重要的文本字段
        text_columns = ['标题 (中文)', '摘要 (中文)', '首项权利要求', 
                       '技术功效句', '用途']
        
        for idx, row in self.data.iterrows():
            combined_text = ""
            for col in text_columns:
                if col in self.data.columns and pd.notna(row[col]):
                    combined_text += str(row[col]) + " "
            
            # 如果没有文本，使用其他可用信息
            if not combined_text.strip():
                combined_text = " ".join([str(row[col]) for col in self.data.columns 
                                         if col != '质量标签' and pd.notna(row[col])][:5])
            
            texts.append(combined_text[:2000])  # 限制文本长度
        
        return texts
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 使用RoBERTa分词器
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
            'label': torch.tensor(label, dtype=torch.long)
        }

class AttentionLayer(nn.Module):
    """注意力机制层"""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output, mask=None):
        """
        Args:
            lstm_output: [batch_size, seq_len, hidden_size]
            mask: [batch_size, seq_len]
        Returns:
            weighted_output: [batch_size, hidden_size]
        """
        # 计算注意力分数
        scores = self.attention(lstm_output).squeeze(-1)  # [batch_size, seq_len]
        
        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]
        
        # 加权求和
        weighted_output = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            lstm_output  # [batch_size, seq_len, hidden_size]
        ).squeeze(1)  # [batch_size, hidden_size]
        
        return weighted_output, attention_weights

class RoBERTaBiLSTMAttention(nn.Module):
    """RoBERTa + BiLSTM + Attention 模型"""
    
    def __init__(self, num_classes=3, lstm_hidden_size=256, lstm_layers=2, 
                 dropout_rate=0.3, roberta_model_name='hfl/chinese-roberta-wwm-ext'):
        """
        初始化模型
        Args:
            num_classes: 分类数量（A/B/C = 3）
            lstm_hidden_size: LSTM隐藏层大小
            lstm_layers: LSTM层数
            dropout_rate: Dropout率
            roberta_model_name: RoBERTa模型名称
        """
        super(RoBERTaBiLSTMAttention, self).__init__()
        
        # 加载预训练的RoBERTa模型
        print(f"加载RoBERTa模型: {roberta_model_name}")
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.roberta_hidden_size = self.roberta.config.hidden_size
        
        # 冻结RoBERTa部分层（可选，加快训练）
        for param in self.roberta.embeddings.parameters():
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
        
        # Attention层
        self.attention = AttentionLayer(lstm_hidden_size * 2)  # *2 因为是双向
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            logits: [batch_size, num_classes]
        """
        # RoBERTa编码
        roberta_outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = roberta_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # BiLSTM处理
        lstm_output, _ = self.bilstm(sequence_output)  # [batch_size, seq_len, hidden_size*2]
        
        # Attention机制
        attended_output, attention_weights = self.attention(lstm_output, attention_mask)
        
        # Dropout
        attended_output = self.dropout(attended_output)
        
        # 分类
        logits = self.classifier(attended_output)
        
        return logits

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
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
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, all_predictions, all_labels

def main():
    """主函数"""
    print("=" * 60)
    print("RoBERTa + BiLSTM + Attention 专利质量分级模型")
    print("=" * 60)
    
    # 超参数设置
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 10
    MAX_LENGTH = 256
    
    # 加载分词器
    print("\n1. 加载RoBERTa分词器...")
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    
    # 创建数据集
    print("\n2. 加载数据集...")
    train_dataset = PatentDataset('train_data.xlsx', tokenizer, max_length=MAX_LENGTH)
    test_dataset = PatentDataset('test_data.xlsx', tokenizer, max_length=MAX_LENGTH, is_test=True)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    print("\n3. 初始化模型...")
    model = RoBERTaBiLSTMAttention(
        num_classes=3,
        lstm_hidden_size=256,
        lstm_layers=2,
        dropout_rate=0.3
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 训练模型
    print("\n4. 开始训练...")
    print("-" * 60)
    
    best_accuracy = 0
    for epoch in range(EPOCHS):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
        
        # 调整学习率
        scheduler.step(test_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"  训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"  测试 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        
        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✓ 保存最佳模型 (Accuracy: {best_accuracy:.4f})")
        
        print("-" * 60)
    
    # 加载最佳模型进行最终评估
    print("\n5. 最终评估...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
    
    # 生成分类报告
    label_names = ['A', 'B', 'C']
    print("\n分类报告:")
    print(classification_report(labels, predictions, target_names=label_names))
    
    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    print("\n混淆矩阵:")
    print("预测→  A    B    C")
    for i, label in enumerate(label_names):
        print(f"{label}    {cm[i]}")
    
    print("\n" + "=" * 60)
    print(f"训练完成! 最佳测试准确率: {best_accuracy:.4f}")
    print("=" * 60)
    
    # 保存结果
    results = {
        'best_accuracy': best_accuracy,
        'classification_report': classification_report(labels, predictions, target_names=label_names, output_dict=True),
        'confusion_matrix': cm.tolist()
    }
    
    import json
    with open('model_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n模型和结果已保存:")
    print("  - best_model.pth (模型权重)")
    print("  - model_results.json (评估结果)")

if __name__ == "__main__":
    main()
