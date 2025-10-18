#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoBERTa + BiLSTM + Cross-Attention 模型实现
使用Cross Attention融合文本特征和数值特征
用于专利质量分级（A/B/C）
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
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class PatentDatasetWithFeatures(Dataset):
    """专利数据集类（包含文本和数值特征）"""
    
    def __init__(self, data_path, tokenizer, max_length=256, is_test=False):
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
        
        print(f"加载数据: {data_path}")
        print(f"数据来源: 真实专利数据 merged_data(1)_cleaned.xlsx")
        print(f"数据量: {len(self.data)}")
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(['A', 'B', 'C'])
        self.labels = self.label_encoder.transform(self.data['质量标签'])
        
        # 准备文本数据
        self.texts = self._prepare_texts()
        
        # 准备数值特征（排除被引用相关信息）
        self.numerical_features = self._prepare_numerical_features()
        
        print(f"标签类别: {self.label_encoder.classes_}")
        print(f"数值特征维度: {self.numerical_features.shape[1]}")
    
    def _prepare_texts(self):
        """准备文本数据，组合多个相关字段"""
        texts = []
        
        # 选择重要的文本字段
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
        """
        准备数值特征（不包含专利被引用相关信息）
        """
        # 选择数值特征列（排除被引用相关的列）
        numerical_columns = [
            '权利要求数量',           # 权利要求的数量
            '首权字数',               # 首项权利要求的字数
            '申请人数量',             # 申请人数量
            '发明(设计)人数量',       # 发明人数量
            '简单同族个数',           # 简单同族专利数
            '扩展同族个数',           # 扩展同族专利数
            'DocDB同族个数'           # DocDB同族专利数
            # 注意：不包含 '家族被引证次数', '家族引证次数' 等被引用相关信息
        ]
        
        features = []
        valid_columns = []
        
        for col in numerical_columns:
            if col in self.data.columns:
                # 转换为数值类型，填充缺失值
                col_data = pd.to_numeric(self.data[col], errors='coerce')
                col_data = col_data.fillna(col_data.median() if col_data.notna().any() else 0)
                features.append(col_data.values.reshape(-1, 1))
                valid_columns.append(col)
        
        print(f"使用的数值特征: {valid_columns}")
        
        if features:
            features = np.hstack(features)
            # 标准化
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        else:
            # 如果没有数值特征，创建零向量
            features = np.zeros((len(self.data), 1))
        
        return features.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        numerical_feat = self.numerical_features[idx]
        
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
            'numerical_features': torch.tensor(numerical_feat, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

class CrossAttention(nn.Module):
    """Cross Attention机制，用于融合文本和数值特征"""
    
    def __init__(self, text_dim, num_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        
        # 投影层
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.num_proj = nn.Linear(num_dim, hidden_dim)
        
        # 注意力计算
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = hidden_dim ** 0.5
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
        
        # 残差连接
        fused_features = attended_features + num_proj.squeeze(1)
        
        return fused_features, attention_weights.squeeze(1)

class RoBERTaBiLSTMCrossAttention(nn.Module):
    """RoBERTa + BiLSTM + Cross-Attention 模型"""
    
    def __init__(self, num_classes=3, lstm_hidden_size=256, lstm_layers=2, 
                 numerical_dim=7, fusion_hidden_dim=256, dropout_rate=0.3):
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
        super(RoBERTaBiLSTMCrossAttention, self).__init__()
        
        # 加载预训练的RoBERTa模型
        roberta_model_name = 'hfl/chinese-roberta-wwm-ext'
        print(f"加载RoBERTa模型: {roberta_model_name}")
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.roberta_hidden_size = self.roberta.config.hidden_size
        
        # 冻结RoBERTa部分层
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
        
        # Cross Attention层（融合文本和数值特征）
        self.cross_attention = CrossAttention(
            text_dim=lstm_hidden_size * 2,  # BiLSTM输出
            num_dim=numerical_dim,
            hidden_dim=fusion_hidden_dim
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
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            numerical_features: [batch_size, numerical_dim]
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
        
        # Cross Attention融合文本和数值特征
        fused_features, attention_weights = self.cross_attention(lstm_output, numerical_features)
        
        # Dropout
        fused_features = self.dropout(fused_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
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
        numerical_features = batch['numerical_features'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, numerical_features)
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

def main():
    """主函数"""
    print("=" * 60)
    print("RoBERTa + BiLSTM + Cross-Attention 专利质量分级模型")
    print("数据来源: 真实专利数据 merged_data(1)_cleaned.xlsx")
    print("=" * 60)
    
    # 超参数设置
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 10
    MAX_LENGTH = 256
    
    # 加载分词器
    print("\n1. 加载RoBERTa分词器...")
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    
    # 创建数据集（使用清理后的数据，无引证特征）
    print("\n2. 加载数据集（已去除引证特征）...")
    train_dataset = PatentDatasetWithFeatures('train_data_clean.xlsx', tokenizer, max_length=MAX_LENGTH)
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
        optimizer, mode='min', factor=0.5, patience=2
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
            torch.save(model.state_dict(), 'best_crossattention_model.pth')
            print(f"  ✓ 保存最佳模型 (Accuracy: {best_accuracy:.4f})")
        
        print("-" * 60)
    
    # 加载最佳模型进行最终评估
    print("\n5. 最终评估...")
    model.load_state_dict(torch.load('best_crossattention_model.pth'))
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
    with open('crossattention_model_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n模型和结果已保存:")
    print("  - best_crossattention_model.pth (模型权重)")
    print("  - crossattention_model_results.json (评估结果)")

if __name__ == "__main__":
    main()
