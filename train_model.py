#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的模型训练脚本 - 用于快速测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class SimplePatentDataset(Dataset):
    """简化的专利数据集类"""
    
    def __init__(self, data_path, tokenizer, max_length=128):
        self.data = pd.read_excel(data_path).head(100)  # 只使用前100条数据进行快速测试
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(['A', 'B', 'C'])
        self.labels = self.label_encoder.transform(self.data['质量标签'])
        
        # 准备文本数据
        self.texts = self._prepare_texts()
    
    def _prepare_texts(self):
        texts = []
        text_columns = ['标题 (中文)', '摘要 (中文)']
        
        for idx, row in self.data.iterrows():
            combined_text = ""
            for col in text_columns:
                if col in self.data.columns and pd.notna(row[col]):
                    combined_text += str(row[col]) + " "
            
            if not combined_text.strip():
                combined_text = "专利文本"
            
            texts.append(combined_text[:500])
        
        return texts
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
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

class SimplifiedRoBERTaModel(nn.Module):
    """简化的RoBERTa + BiLSTM + Attention模型"""
    
    def __init__(self, num_classes=3):
        super(SimplifiedRoBERTaModel, self).__init__()
        
        # 使用中文RoBERTa模型
        self.roberta = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        hidden_size = self.roberta.config.hidden_size
        
        # 冻结大部分层以加快训练
        for name, param in self.roberta.named_parameters():
            if 'pooler' not in name and 'encoder.layer.11' not in name:
                param.requires_grad = False
        
        # BiLSTM
        self.lstm = nn.LSTM(
            hidden_size, 
            128, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 分类器
        self.classifier = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, input_ids, attention_mask):
        # RoBERTa编码
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # BiLSTM
        lstm_out, _ = self.lstm(sequence_output)
        
        # 使用最后一个时间步的输出
        pooled = lstm_out[:, -1, :]
        pooled = self.dropout(pooled)
        
        # 分类
        logits = self.classifier(pooled)
        
        return logits

def quick_train():
    """快速训练测试"""
    print("=" * 60)
    print("RoBERTa + BiLSTM 模型快速测试")
    print("=" * 60)
    
    # 参数设置
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    EPOCHS = 3  # 只训练3个epoch进行测试
    MAX_LENGTH = 128
    
    try:
        # 加载分词器
        print("\n1. 加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        
        # 创建数据集
        print("\n2. 加载数据集（测试版本）...")
        train_dataset = SimplePatentDataset('train_data.xlsx', tokenizer, MAX_LENGTH)
        test_dataset = SimplePatentDataset('test_data.xlsx', tokenizer, MAX_LENGTH)
        
        print(f"   训练集大小: {len(train_dataset)}")
        print(f"   测试集大小: {len(test_dataset)}")
        
        # 数据加载器
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 初始化模型
        print("\n3. 初始化模型...")
        model = SimplifiedRoBERTaModel(num_classes=3).to(device)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        
        # 训练
        print("\n4. 开始训练...")
        print("-" * 60)
        
        for epoch in range(EPOCHS):
            # 训练阶段
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # 评估阶段
            model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            train_acc = train_correct / train_total
            test_acc = test_correct / test_total
            
            print(f"Epoch [{epoch+1}/{EPOCHS}]")
            print(f"  训练 - Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_acc:.4f}")
            print(f"  测试 - Accuracy: {test_acc:.4f}")
            print("-" * 60)
        
        # 保存模型
        torch.save(model.state_dict(), 'quick_test_model.pth')
        print("\n✓ 模型训练完成!")
        print("✓ 模型已保存到: quick_test_model.pth")
        
        return True
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_train()
    if success:
        print("\n" + "=" * 60)
        print("快速测试成功! 模型架构可以正常工作。")
        print("您可以运行 roberta_bilstm_model.py 进行完整训练。")
        print("=" * 60)
