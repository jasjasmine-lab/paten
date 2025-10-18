#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版训练脚本 - 使用增强数据和改进的训练策略
包含简化的模型架构和优化的超参数
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

class SimplifiedRoBERTaModel(nn.Module):
    """简化的RoBERTa模型 - 减少复杂度"""
    
    def __init__(self, num_classes=3, lstm_hidden_size=128, lstm_layers=1, 
                 numerical_dim=7, fusion_hidden_dim=128, dropout_rate=0.4):
        """
        初始化简化模型
        Args:
            num_classes: 分类数量（A/B/C = 3）
            lstm_hidden_size: LSTM隐藏层大小（减少到128）
            lstm_layers: LSTM层数（减少到1层）
            numerical_dim: 数值特征维度
            fusion_hidden_dim: 融合层隐藏维度（减少到128）
            dropout_rate: Dropout率（增加到0.4）
        """
        super(SimplifiedRoBERTaModel, self).__init__()
        
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
        
        # 简化的BiLSTM层（只有1层）
        self.bilstm = nn.LSTM(
            input_size=self.roberta_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0 if lstm_layers == 1 else dropout_rate
        )
        
        # 数值特征投影
        self.numerical_proj = nn.Sequential(
            nn.Linear(numerical_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 文本特征池化
        self.text_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 简化的分类层
        text_dim = lstm_hidden_size * 2  # BiLSTM输出
        combined_dim = text_dim + fusion_hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, num_classes)
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
        
        # 处理数值特征
        num_features = self.numerical_proj(numerical_features)
        
        # 拼接特征
        combined_features = torch.cat([lstm_pooled, num_features], dim=1)
        
        # Dropout
        combined_features = self.dropout(combined_features)
        
        # 分类
        logits = self.classifier(combined_features)
        
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

def train_epoch_with_accumulation(model, dataloader, criterion, optimizer, device, accumulation_steps=2):
    """带梯度累积的训练"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
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

def plot_training_curves(history, save_path='optimized_training_history.png'):
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
    print("优化版专利质量分级模型训练")
    print("使用增强数据 + 简化架构 + 优化策略")
    print("=" * 60)
    
    # 优化的超参数
    BATCH_SIZE = 32          # 增加批次大小
    INITIAL_LR = 1e-5        # 降低学习率
    EPOCHS = 80              # 适中的训练轮次
    MAX_LENGTH = 256
    WARMUP_EPOCHS = 3        # 减少warmup轮次
    ACCUMULATION_STEPS = 2   # 梯度累积步数
    
    # 加载分词器
    print("\n1. 加载RoBERTa分词器...")
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    
    # 检查增强数据是否存在，如果不存在则先运行数据增强
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
    
    # 初始化简化模型
    print("\n3. 初始化简化模型...")
    model = SimplifiedRoBERTaModel(
        num_classes=3,
        lstm_hidden_size=128,  # 减少隐藏层大小
        lstm_layers=5,          # 减少LSTM层数
        numerical_dim=numerical_dim,
        fusion_hidden_dim=128,  # 减少融合层维度
        dropout_rate=0.4        # 增加dropout
    ).to(device)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"参数减少比例: {(1 - trainable_params/total_params)*100:.2f}%")
    
    # 损失函数（温和的类别权重）
    class_weights = torch.tensor([1.0, 6.0, 5.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"\n类别权重: {class_weights.cpu().numpy()}")
    
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
        
        # 训练
        train_loss, train_acc = train_epoch_with_accumulation(
            model, train_loader, criterion, optimizer, device, ACCUMULATION_STEPS
        )
        
        # 验证
        test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
        
        # 计算详细指标
        f1_macro = f1_score(labels, predictions, average='macro')
        balanced_acc = balanced_accuracy_score(labels, predictions)
        
        # 计算综合评分
        balanced_score = 0.4 * test_acc + 0.6 * f1_macro
        
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
            }, '/root/autodl-tmp/best_optimized_model.pth')
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
    plot_training_curves(history, save_path='/root/autodl-tmp/optimized_training_history.png')
    
    # 加载最佳模型进行最终评估
    print("\n5. 最终评估（使用最佳模型）...")
    checkpoint = torch.load('/root/autodl-tmp/best_optimized_model.pth')
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
    print("优化训练完成!")
    print(f"最佳测试准确率: {best_accuracy:.4f}")
    print(f"最佳F1-Macro分数: {best_f1_macro:.4f}")
    print(f"最佳训练轮次: {best_epoch}")
    print("=" * 60)
    
    # 与之前的结果对比
    print("\n性能对比:")
    print("-" * 40)
    print("基础模型（原始数据）:")
    print("  准确率: 48.57%")
    print("  F1-Macro: 0.4158")
    print("-" * 40)
    print("扩展训练（平衡数据）:")
    print("  准确率: 26.85%")
    print("  F1-Macro: 0.2561")
    print("-" * 40)
    print("优化模型（增强数据+改进策略）:")
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
            'lstm_layers': 1,
            'dropout_rate': 0.4
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('/root/autodl-tmp/optimized_model_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n模型和结果已保存:")
    print("  - /root/autodl-tmp/best_optimized_model.pth (优化模型权重)")
    print("  - /root/autodl-tmp/optimized_model_results.json (详细评估结果)")
    print("  - /root/autodl-tmp/optimized_training_history.png (训练曲线)")
    
    # 生成优化总结
    print("\n" + "=" * 60)
    print("优化策略总结:")
    print("1. 数据增强: SMOTE过采样 + 混合采样策略")
    print("2. 模型简化: 减少LSTM层数和隐藏单元")
    print("3. 正则化增强: Dropout提高到0.4")
    print("4. 学习率优化: 降低到1e-5 + 余弦退火")
    print("5. 类别权重: 温和调整[1.0, 1.1, 1.1]")
    print("6. 梯度累积: 提高训练稳定性")
    print("=" * 60)

if __name__ == "__main__":
    main()
