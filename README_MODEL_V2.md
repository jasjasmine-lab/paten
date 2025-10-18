# 专利质量分级模型 V2 使用说明

## 模型改进

### 主要改进点
1. **长文本支持**: 从256 tokens增加到2048 tokens
2. **全面文本拼接**: 整合所有可用文本字段
3. **时间标准化评分**: 使用 `被引次数 / (min(到期日, 当前日期) - 公开日)` 公式
4. **均衡标签分布**: 确保A/B/C三类各占约33.33%
5. **梯度累积**: 处理GPU内存限制

## 快速开始

### 1. 环境要求
- Python 3.8+
- CUDA 11.0+ (推荐，用于GPU加速)
- 至少8GB GPU显存（推荐16GB）
- 约50GB磁盘空间

### 2. 安装依赖
```bash
pip install pandas numpy torch transformers scikit-learn tqdm openpyxl joblib
```

### 3. 准备数据
确保已有 `merged_patents_inner.xlsx` 文件（通过运行合并脚本生成）

### 4. 运行完整流程
```bash
# 运行完整流程（数据预处理 + 模型训练）
python run_pipeline.py

# 只运行数据预处理
python run_pipeline.py --only-prepare

# 跳过数据预处理，直接训练
python run_pipeline.py --skip-data-prep

# 自定义训练参数
python run_pipeline.py --epochs 5 --batch-size 1
```

## 分步执行

### 步骤1: 数据预处理
```bash
python prepare_patent_data.py
```

这会生成：
- `train_data_prepared.xlsx` - 训练集（80%）
- `test_data_prepared.xlsx` - 测试集（20%）
- `all_data_prepared.xlsx` - 完整数据集
- `numerical_scaler.pkl` - 数值特征标准化器
- `feature_names.txt` - 特征名称列表

### 步骤2: 模型训练
```bash
python roberta_bilstm_crossattention_model_v2.py
```

这会生成：
- `best_model_v2.pth` - 最佳模型权重
- `checkpoint_epoch_*.pth` - 各轮次检查点
- `model_results_v2.json` - 训练结果和评估指标

## 数据处理细节

### 质量评分计算
```python
质量评分 = 被引次数 / ((min(到期日, 当前日期) - 公开日) / 365)
```
- 消除专利年龄对引证次数的影响
- 公平比较不同年代的专利

### 文本拼接策略
拼接以下字段（使用结构化标记）：
- `[标题]` 标题 (中文)
- `[摘要]` 摘要 (中文)
- `[权利要求]` 首项权利要求
- `[技术功效]` 技术功效
- `[技术手段]` 技术手段
- `[功效描述]` 技术功效句
- `[应用领域]` 用途

缺失字段处理：
- 技术功效句缺失时用技术功效替代
- 其他字段缺失时跳过

### 数值特征（7维）
1. 权利要求数量
2. 首权字数
3. 申请人数量
4. 发明人数量
5. 简单同族个数
6. 扩展同族个数
7. DocDB同族个数

**注意**: 不包含引证信息，避免数据泄露

## 模型架构

### RoBERTa + BiLSTM + Cross-Attention
```
输入文本 (2048 tokens) → RoBERTa → BiLSTM → 
                                              ↘
                                          Cross-Attention → 分类器 → A/B/C
                                              ↗
输入数值特征 (7维) → 标准化 → 投影层 →
```

### 关键参数
- **RoBERTa**: chinese-roberta-wwm-ext
- **BiLSTM**: 256维隐藏层，2层
- **Cross-Attention**: 256维融合层
- **Dropout**: 0.3
- **批次大小**: 2 (有效16，通过梯度累积)
- **学习率**: 2e-5

## 性能优化

### GPU内存不足解决方案
1. **减小批次大小**
   ```bash
   python run_pipeline.py --batch-size 1
   ```

2. **减少最大序列长度**
   修改 `roberta_bilstm_crossattention_model_v2.py`:
   ```python
   MAX_LENGTH = 1024  # 从2048改为1024
   ```

3. **使用混合精度训练**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   with autocast():
       outputs = model(...)
   ```

### 训练时间优化
1. **减少训练轮数**
   ```bash
   python run_pipeline.py --epochs 5
   ```

2. **使用更大的批次**（如果显存允许）
   ```bash
   python run_pipeline.py --batch-size 4
   ```

## 预期性能

| 指标 | 预期值 | 说明 |
|-----|--------|------|
| 准确率 | 83-85% | 测试集准确率 |
| F1分数 | ~0.82 | 宏平均F1 |
| 训练时间 | 2-3小时/epoch | 使用GPU |
| 显存使用 | 6-8GB | batch_size=2 |

## 常见问题

### Q1: OOM (Out of Memory) 错误
**解决方案**:
- 减小batch_size到1
- 减小MAX_LENGTH到1024
- 确保没有其他进程占用GPU

### Q2: 训练速度很慢
**解决方案**:
- 确认使用GPU而非CPU
- 检查数据加载是否成为瓶颈
- 考虑减少训练轮数

### Q3: 标签分布不均衡
**解决方案**:
- 重新运行数据预处理
- 检查质量评分计算是否正确
- 确认使用33.33%和66.67%分位数

### Q4: 模型不收敛
**解决方案**:
- 降低学习率到1e-5
- 增加训练轮数
- 检查数据质量

## 模型评估

### 查看训练结果
```python
import json

# 读取结果
with open('model_results_v2.json', 'r') as f:
    results = json.load(f)

# 打印最佳准确率
print(f"最佳准确率: {results['best_accuracy']:.4f}")

# 打印分类报告
for label in ['A', 'B', 'C']:
    metrics = results['classification_report'][label]
    print(f"{label}级 - Precision: {metrics['precision']:.3f}, "
          f"Recall: {metrics['recall']:.3f}, "
          f"F1: {metrics['f1-score']:.3f}")

# 打印混淆矩阵
print("\n混淆矩阵:")
cm = results['confusion_matrix']
print("预测→  A    B    C")
for i, label in enumerate(['A', 'B', 'C']):
    print(f"{label}    {cm[i]}")
```

## 模型使用

### 加载模型进行预测
```python
import torch
from transformers import AutoTokenizer
import joblib
import pandas as pd

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RoBERTaBiLSTMCrossAttentionV2().to(device)
checkpoint = torch.load('best_model_v2.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 加载分词器和标准化器
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
scaler = joblib.load('numerical_scaler.pkl')

# 准备输入数据
def predict_patent_quality(patent_data):
    # 准备文本
    text = prepare_text(patent_data)
    encoding = tokenizer(text, truncation=True, padding='max_length', 
                        max_length=2048, return_tensors='pt')
    
    # 准备数值特征
    numerical_features = prepare_numerical_features(patent_data)
    numerical_features = scaler.transform(numerical_features.reshape(1, -1))
    
    # 预测
    with torch.no_grad():
        outputs = model(
            encoding['input_ids'].to(device),
            encoding['attention_mask'].to(device),
            torch.tensor(numerical_features, dtype=torch.float32).to(device)
        )
        _, predicted = torch.max(outputs, 1)
    
    labels = ['A', 'B', 'C']
    return labels[predicted.item()]
```

## 项目结构
```
autodl-tmp/
├── prepare_patent_data.py           # 数据预处理脚本
├── roberta_bilstm_crossattention_model_v2.py  # 模型训练脚本
├── run_pipeline.py                  # 完整流程脚本
├── README_MODEL_V2.md               # 本文档
│
├── merged_patents_inner.xlsx        # 输入：内连接数据
├── train_data_prepared.xlsx         # 输出：训练数据
├── test_data_prepared.xlsx          # 输出：测试数据
├── best_model_v2.pth               # 输出：最佳模型
├── model_results_v2.json           # 输出：评估结果
├── numerical_scaler.pkl            # 输出：特征标准化器
└── checkpoint_epoch_*.pth          # 输出：训练检查点
```

## 更新日志

### V2.0 (2025-10-16)
- 增加最大token数到2048
- 实现全文本字段拼接
- 采用时间标准化质量评分
- 确保标签均衡分布
- 添加梯度累积支持
- 优化内存使用

## 联系与支持

如有问题或建议，请参考：
- 检查常见问题部分
- 查看错误日志
- 验证数据格式是否正确

## 许可证

本项目仅供研究使用。
