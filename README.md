# 专利质量分级系统
Patent Quality Classification System

## 项目简介

基于深度学习的专利质量自动分级系统，使用 RoBERTa + BiLSTM + Cross-Attention + ResNet 架构，实现对专利的A、B、C三级质量分类。

## 主要特性

- **多模态融合**：结合文本特征和数值特征进行综合评估
- **分块处理**：支持长文本专利的完整处理
- **注意力机制**：使用Cross-Attention融合不同类型特征
- **ResNet增强**：通过残差网络增强特征表示

## 模型架构

### V3版本 (ResNet增强版)
- **文件**: `chunked_roberta_v3_resnet.py`
- **特点**:
  - RoBERTa中文预训练模型 (hfl/chinese-roberta-wwm-ext)
  - BiLSTM (384维隐藏层 × 2层)
  - ResNet模块 (文本3块 + 融合2块)
  - 8头Cross-Attention机制
  - 块大小: 512 tokens，重叠: 128 tokens

### V2版本 (基础版)
- **文件**: `chunked_roberta_v2.py`
- **特点**:
  - 基础Cross-Attention架构
  - 更轻量级的模型设计

## 数据处理流程

1. **数据准备** (`prepare_patent_data.py`)
   - 专利文本预处理
   - 数值特征提取
   - 标签编码

2. **数据质量验证**
   - `verify_train_test_split.py` - 训练集测试集分割验证
   - `verify_no_label_leakage.py` - 标签泄露检查
   - `verify_processed_data.py` - 数据完整性验证

3. **数据分析**
   - `analyze_merged_data_quality.py` - 数据质量分析
   - `analyze_token_usage.py` - Token使用统计

## 使用方法

### 环境要求
```bash
- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- CUDA 11.0+ (推荐)
```

### 安装依赖
```bash
pip install torch transformers pandas numpy scikit-learn tqdm
```

### 训练模型
```bash
# V3版本 (ResNet增强)
python chunked_roberta_v3_resnet.py

# V2版本
python chunked_roberta_v2.py
```

### 测试模型
```bash
python test_model_v3.py
```

### 完整流程运行
```bash
python run_pipeline.py
```

## 模型优化建议

### 可扩展参数层
1. **BiLSTM层**: 384 → 512/768维
2. **Cross-Attention**: 增加注意力头数 8 → 12/16
3. **ResNet块**: 增加深度 3 → 5-6块
4. **FFN层**: 扩展隐藏层倍数 4x → 6x/8x

### 优化方案
- **方案A (平衡型)**: 参数增量~5-6M，预期提升3-5%
- **方案B (深度型)**: 参数增量~10-12M，预期提升5-8%

## 项目结构

```
├── 模型代码/
│   ├── chunked_roberta_v3_resnet.py    # V3模型主文件
│   ├── chunked_roberta_v2.py           # V2模型
│   └── roberta_bilstm_crossattention_model_v2.py  # 基础模型
├── 数据处理/
│   ├── prepare_patent_data.py          # 数据准备
│   ├── merge_patent_data_by_publication_number.py  # 数据合并
│   └── split_dataset.py                # 数据集分割
├── 验证脚本/
│   ├── verify_train_test_split.py      # 数据分割验证
│   ├── verify_no_label_leakage.py      # 标签泄露检查
│   └── test_model_v3.py                # 模型测试
└── 文档/
    ├── README_MODEL_V3_RESNET.md       # V3模型说明
    └── README_MODEL_V2.md               # V2模型说明
```

## 性能指标

- **准确率**: ~85-90% (取决于数据质量和模型配置)
- **训练时间**: 约2-3小时/100轮 (GPU)
- **显存占用**: 8-12GB (批次大小=8)

## 注意事项

1. 确保有足够的显存运行模型
2. 数据文件(xlsx, csv)已在.gitignore中排除
3. 模型权重文件(.pth)不包含在仓库中
4. 建议使用GPU进行训练

## 作者

jasjasmine-lab

## License

本项目仅供学习研究使用

## 更新日志

- 2024.10.18: 初始版本发布
  - 实现RoBERTa + BiLSTM + Cross-Attention + ResNet架构
  - 支持专利质量三级分类
  - 完整的数据处理和验证流程
