#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoBERTa + BiLSTM + Cross-Attention 模型实现 V2
改进版本：
1. 支持2048 tokens长文本输入
2. 拼接所有文本字段
3. 使用时间标准化的质量评分
4. 均衡的A/B/C标签
5. 梯度累积处理小批次
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
import joblib
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 如果使用GPU，打印显存信息
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

class PatentDatasetV2(Dataset):
    """专利数据集类V2（支持分块处理长文本）"""
    
    def __init__(self, data_path, tokenizer, chunk_size=512, overlap=128, max_chunks=16, is_test=False):
        """
        初始化数据集
        Args:
            data_path: Excel文件路径
            tokenizer: RoBERTa分词器
            chunk_size: 每个块的最大长度（默认512）
            overlap: 块之间的重叠token数（默认128）
            max_chunks: 最大块数（默认16）
            is_test: 是否为测试集
        """
        self.data = pd.read_excel(data_path)
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = chunk_size - overlap
        self.max_chunks = max_chunks
        self.is_test = is_test
        
        print(f"\n加载数据: {data_path}")
        print(f"  数据量: {len(self.data):,}")
        print(f"  块大小: {chunk_size} tokens")
        print(f"  重叠: {overlap} tokens")
        print(f"  最大块数: {max_chunks}")
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(['A', 'B', 'C'])
        self.labels = self.label_encoder.transform(self.data['质量标签'])
        
        # 获取文本数据（已经拼接好的）
        self.texts = self.data['组合文本'].tolist()
        
        # 准备数值特征
        self.numerical_features = self._prepare_numerical_features()
        
        # 打印统计信息
        print(f"  标签类别: {self.label_encoder.classes_}")
        label_counts = pd.Series(self.labels).value_counts().sort_index()
        for i, label in enumerate(self.label_encoder.classes_):
            count = label_counts.get(i, 0)
            print(f"    {label}级: {count:,}条 ({count/len(self.data)*100:.1f}%)")
        print(f"  数值特征维度: {self.numerical_features.shape[1]}")
    
    def _prepare_numerical_features(self):
        """
        从DataFrame中提取已处理的数值特征
        """
        # 获取所有数值特征列
        feature_columns = [col for col in self.data.columns if col.startswith('数值特征_')]
        
        if not feature_columns:
            print("  警告：未找到数值特征列，使用默认值")
            return np.zeros((len(self.data), 7), dtype=np.float32)
        
        features = self.data[feature_columns].values.astype(np.float32)
        
        # 检查NaN值
        if np.any(np.isnan(features)):
            print("  警告：数值特征中存在NaN，进行填充")
            features = np.nan_to_num(features, 0)
        
        return features
    
    def __len__(self):
        return len(self.data)
    
    def _create_chunks(self, text):
        """
        将长文本分成多个块
        Returns:
            input_ids_list: 每个块的input_ids
            attention_mask_list: 每个块的attention_mask
        """
        # 先进行初步分词
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # 创建块
        chunks_input_ids = []
        chunks_attention_mask = []
        
        # 滑动窗口分块
        for i in range(0, len(tokens), self.stride):
            # 提取块的tokens
            chunk_tokens = tokens[i:i + self.chunk_size - 2]  # 留空间给[CLS]和[SEP]
            
            # 如果块太短，跳过
            if len(chunk_tokens) < 10:
                continue
            
            # 编码这个块
            encoding = self.tokenizer.encode_plus(
                self.tokenizer.decode(chunk_tokens),
                truncation=True,
                padding='max_length',
                max_length=self.chunk_size,
                return_tensors='pt'
            )
            
            chunks_input_ids.append(encoding['input_ids'].squeeze(0))
            chunks_attention_mask.append(encoding['attention_mask'].squeeze(0))
            
            # 限制最大块数
            if len(chunks_input_ids) >= self.max_chunks:
                break
            
            # 如果已经覆盖全部文本，停止
            if i + self.chunk_size >= len(tokens):
                break
        
        # 如果没有块，创建一个默认块
        if len(chunks_input_ids) == 0:
            encoding = self.tokenizer.encode_plus(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.chunk_size,
                return_tensors='pt'
            )
            chunks_input_ids.append(encoding['input_ids'].squeeze(0))
            chunks_attention_mask.append(encoding['attention_mask'].squeeze(0))
        
        # 填充到固定数量的块
        while len(chunks_input_ids) < self.max_chunks:
            # 添加空块（全padding）
            empty_input_ids = torch.zeros(self.chunk_size, dtype=torch.long)
            empty_attention_mask = torch.zeros(self.chunk_size, dtype=torch.long)
            chunks_input_ids.append(empty_input_ids)
            chunks_attention_mask.append(empty_attention_mask)
        
        # Stack成tensor
        input_ids = torch.stack(chunks_input_ids[:self.max_chunks])
        attention_mask = torch.stack(chunks_attention_mask[:self.max_chunks])
        
        return input_ids, attention_mask
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        numerical_feat = self.numerical_features[idx]
        
        # 创建文本块
        input_ids, attention_mask = self._create_chunks(text)
        
        return {
            'input_ids': input_ids,  # [max_chunks, chunk_size]
            'attention_mask': attention_mask,  # [max_chunks, chunk_size]
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

class RoBERTaBiLSTMCrossAttentionV2(nn.Module):
    """RoBERTa + BiLSTM + Cross-Attention 模型 V2"""
    
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
        super(RoBERTaBiLSTMCrossAttentionV2, self).__init__()
        
        # 加载预训练的RoBERTa模型
        roberta_model_name = 'hfl/chinese-roberta-wwm-ext'
        print(f"加载RoBERTa模型: {roberta_model_name}")
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.roberta_hidden_size = self.roberta.config.hidden_size
        
        # 冻结RoBERTa部分层（只冻结embedding层，允许其他层微调）
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

def train_epoch_with_accumulation(model, dataloader, criterion, optimizer, device, 
                                 accumulation_steps=8, scheduler=None):
    """
    使用梯度累积的训练
    Args:
        accumulation_steps: 梯度累积步数（用于模拟更大的batch size）
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 初始化梯度
    optimizer.zero_grad()
    
    # 进度条
    progress_bar = tqdm(dataloader, desc="Training")
    
    for i, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        numerical_features = batch['numerical_features'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        outputs = model(input_ids, attention_mask, numerical_features)
        loss = criterion(outputs, labels)
        
        # 梯度累积：将损失除以累积步数
        loss = loss / accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # 每accumulation_steps步进行一次参数更新
        if (i + 1) % accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 参数更新
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            # 清零梯度
            optimizer.zero_grad()
        
        # 统计
        total_loss += loss.item() * accumulation_steps  # 恢复原始损失值
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    # 处理最后的梯度（如果有剩余）
    if len(dataloader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    # 进度条
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
            
            # 更新进度条
            current_acc = accuracy_score(all_labels, all_predictions)
            progress_bar.set_postfix({'acc': f'{current_acc:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, all_predictions, all_labels

def main():
    """主函数"""
    print("=" * 80)
    print("RoBERTa + BiLSTM + Cross-Attention 专利质量分级模型 V2")
    print("改进：长文本输入 + 时间标准化评分 + 均衡标签")
    print("=" * 80)
    
    # 超参数设置
    BATCH_SIZE = 1          # 进一步减小批次大小以适应超长序列
    ACCUMULATION_STEPS = 16  # 增加梯度累积步数，实际batch size = 1 * 16 = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 50             # 增加到50个epoch进行充分训练
    MAX_LENGTH = 8192       # 扩展到8192 tokens，覆盖所有数据
    
    print("\n超参数设置:")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  梯度累积步数: {ACCUMULATION_STEPS}")
    print(f"  有效批次大小: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  最大序列长度: {MAX_LENGTH}")
    
    # 加载分词器
    print("\n1. 加载RoBERTa分词器...")
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    
    # 创建数据集
    print("\n2. 加载数据集...")
    train_dataset = PatentDatasetV2('train_data_prepared.xlsx', tokenizer, max_length=MAX_LENGTH)
    test_dataset = PatentDatasetV2('test_data_prepared.xlsx', tokenizer, max_length=MAX_LENGTH, is_test=True)
    
    # 获取数值特征维度
    numerical_dim = train_dataset.numerical_features.shape[1]
    print(f"\n数值特征维度: {numerical_dim}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 初始化模型
    print("\n3. 初始化模型...")
    model = RoBERTaBiLSTMCrossAttentionV2(
        num_classes=3,
        lstm_hidden_size=256,
        lstm_layers=2,
        numerical_dim=numerical_dim,
        fusion_hidden_dim=256,
        dropout_rate=0.3
    ).to(device)
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {total_params - trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # 学习率调度器
    total_steps = len(train_loader) * EPOCHS // ACCUMULATION_STEPS
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0,
        end_factor=0.1,
        total_iters=total_steps
    )
    
    # 训练模型
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
        train_loss, train_acc = train_epoch_with_accumulation(
            model, train_loader, criterion, optimizer, device, 
            accumulation_steps=ACCUMULATION_STEPS, scheduler=scheduler
        )
        
        # 验证
        test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
        
        # 记录历史
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
            }, 'best_model_v2.pth')
            print(f"✓ 保存最佳模型 (Accuracy: {best_accuracy:.4f})")
        
        # 每个epoch保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'training_history': training_history,
        }, f'checkpoint_epoch_{epoch+1}.pth')
        
        print("-" * 80)
    
    # 加载最佳模型进行最终评估
    print("\n5. 最终评估...")
    checkpoint = torch.load('best_model_v2.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
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
    
    print("\n" + "=" * 80)
    print(f"训练完成! 最佳测试准确率: {best_accuracy:.4f}")
    print("=" * 80)
    
    # 保存结果
    results = {
        'best_accuracy': best_accuracy,
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
            'max_length': MAX_LENGTH
        }
    }
    
    with open('model_results_v2.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n模型和结果已保存:")
    print("  - best_model_v2.pth (最佳模型权重)")
    print("  - checkpoint_epoch_*.pth (各轮次检查点)")
    print("  - model_results_v2.json (评估结果)")

if __name__ == "__main__":
    main()
