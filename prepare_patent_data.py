#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专利数据预处理脚本
用于准备RoBERTa + BiLSTM + CrossAttention模型的训练数据

主要功能：
1. 计算时间标准化的质量评分
2. 生成均衡的A/B/C标签
3. 拼接所有文本字段
4. 准备数值特征
5. 划分训练集和测试集
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def calculate_time_normalized_quality_score(df):
    """
    计算时间标准化的质量评分
    公式：被引次数 / (min(到期日, 当前日期) - 公开日)
    """
    print("\n计算时间标准化质量评分...")
    
    # 当前日期
    current_date = datetime.now()
    
    scores = []
    valid_count = 0
    missing_citation = 0
    missing_dates = 0
    
    for idx, row in df.iterrows():
        # 获取被引次数（缺失值填0）
        citations = row.get('家族被引证次数', 0)
        if pd.isna(citations):
            citations = 0
            missing_citation += 1
        else:
            citations = float(citations)
        
        # 获取日期
        public_date = pd.to_datetime(row.get('公开（公告）日'), errors='coerce')
        expire_date = pd.to_datetime(row.get('预估到期日'), errors='coerce')
        
        # 处理日期缺失
        if pd.isna(public_date):
            # 如果公开日缺失，使用申请日
            public_date = pd.to_datetime(row.get('申请日'), errors='coerce')
            if pd.isna(public_date):
                # 如果还是缺失，给默认值（2015年）
                public_date = datetime(2015, 1, 1)
                missing_dates += 1
        
        if pd.isna(expire_date):
            # 如果到期日缺失，假设20年专利期
            if not pd.isna(public_date):
                expire_date = public_date + pd.Timedelta(days=20*365)
            else:
                expire_date = datetime(2035, 1, 1)
        
        # 计算有效时间跨度（天数）
        effective_end = min(expire_date, current_date)
        time_span = (effective_end - public_date).days
        
        # 避免除零错误，最小时间跨度设为365天
        time_span = max(time_span, 365)
        
        # 计算标准化评分（年均引用率）
        score = citations / (time_span / 365.0)  # 转换为年
        scores.append(score)
        
        if citations > 0:
            valid_count += 1
    
    print(f"  - 有效引证记录: {valid_count}/{len(df)} ({valid_count/len(df)*100:.1f}%)")
    print(f"  - 缺失引证次数: {missing_citation}")
    print(f"  - 缺失日期信息: {missing_dates}")
    print(f"  - 评分范围: {min(scores):.4f} ~ {max(scores):.4f}")
    print(f"  - 平均评分: {np.mean(scores):.4f}")
    print(f"  - 中位数评分: {np.median(scores):.4f}")
    
    return pd.Series(scores)

def generate_balanced_labels(scores):
    """
    基于评分生成均衡的A/B/C标签
    确保每类约占33.33%
    """
    print("\n生成均衡的质量标签...")
    
    # 计算33.33%和66.67%分位数
    percentile_33 = scores.quantile(0.3333)
    percentile_67 = scores.quantile(0.6667)
    
    print(f"  - 33.33%分位数: {percentile_33:.4f}")
    print(f"  - 66.67%分位数: {percentile_67:.4f}")
    
    labels = []
    for score in scores:
        if score >= percentile_67:
            labels.append('A')  # 前33.33%
        elif score >= percentile_33:
            labels.append('B')  # 中间33.33%
        else:
            labels.append('C')  # 后33.33%
    
    # 验证分布
    label_counts = pd.Series(labels).value_counts()
    print("\n标签分布:")
    for label in ['A', 'B', 'C']:
        count = label_counts.get(label, 0)
        percentage = count / len(labels) * 100
        print(f"  {label}级: {count:,}条 ({percentage:.1f}%)")
    
    return labels

def concatenate_all_text_fields(row, max_chars=15000):
    """
    拼接所有可用的文本字段
    使用结构化标记保留字段信息
    """
    sections = []
    
    # 1. 核心字段（必有）
    if pd.notna(row.get('标题 (中文)')):
        sections.append(f"[标题] {row['标题 (中文)']}")
    
    if pd.notna(row.get('首项权利要求')):
        sections.append(f"[权利要求] {row['首项权利要求']}")
    
    # 2. 重要字段（高优先级）
    if pd.notna(row.get('摘要 (中文)')):
        sections.append(f"[摘要] {row['摘要 (中文)']}")
    
    if pd.notna(row.get('技术功效')):
        sections.append(f"[技术功效] {row['技术功效']}")
    
    if pd.notna(row.get('技术手段')):
        sections.append(f"[技术手段] {row['技术手段']}")
    
    # 3. 补充字段（中优先级）
    if pd.notna(row.get('技术功效句')):
        sections.append(f"[功效描述] {row['技术功效句']}")
    elif pd.notna(row.get('技术功效')):
        # 如果技术功效句缺失，用技术功效补充
        sections.append(f"[功效描述] {row['技术功效'][:500]}")  # 限制长度避免重复
    
    if pd.notna(row.get('用途')):
        sections.append(f"[应用领域] {row['用途']}")
    
    # 4. 其他可用文本字段（如果存在）
    additional_fields = [
        ('说明书摘要', 2000),
        ('背景技术', 1500),
        ('发明内容', 2000),
        ('具体实施方式', 1500),
        ('权利要求书', 2000),
        ('技术领域', 500),
        ('有益效果', 1000)
    ]
    
    for field_name, max_len in additional_fields:
        if field_name in row and pd.notna(row[field_name]):
            # 限制每个字段的长度
            content = str(row[field_name])[:max_len]
            sections.append(f"[{field_name}] {content}")
    
    # 拼接所有部分
    full_text = " [SEP] ".join(sections)
    
    # 限制总长度（约15000字符 ≈ 2048 tokens for Chinese text）
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]
    
    # 确保至少有一些内容
    if not full_text.strip():
        full_text = "[空文档] 无可用文本信息"
    
    return full_text

def prepare_numerical_features_no_citations(df):
    """
    准备数值特征（不包含引证信息，避免数据泄露）
    """
    print("\n准备数值特征...")
    
    features_list = []
    feature_names = []
    
    # 1. 权利要求数量
    if '权利要求数' in df.columns:
        claim_count = pd.to_numeric(df['权利要求数'], errors='coerce').fillna(0)
        features_list.append(claim_count.values.reshape(-1, 1))
        feature_names.append('权利要求数量')
    else:
        # 如果没有这个字段，创建默认值
        features_list.append(np.ones((len(df), 1)))
        feature_names.append('权利要求数量')
    
    # 2. 首权字数（计算）
    if '首项权利要求' in df.columns:
        first_claim_length = df['首项权利要求'].astype(str).str.len()
        features_list.append(first_claim_length.values.reshape(-1, 1))
        feature_names.append('首权字数')
    else:
        features_list.append(np.ones((len(df), 1)) * 500)  # 默认值
        feature_names.append('首权字数')
    
    # 3. 申请人数量（分割统计）
    if '申请人' in df.columns:
        applicant_count = df['申请人'].astype(str).str.split(';|；').str.len()
        features_list.append(applicant_count.values.reshape(-1, 1))
        feature_names.append('申请人数量')
    else:
        features_list.append(np.ones((len(df), 1)))
        feature_names.append('申请人数量')
    
    # 4. 发明人数量
    inventor_field = None
    for field in ['发明(设计)人', '发明人', '发明（设计）人']:
        if field in df.columns:
            inventor_field = field
            break
    
    if inventor_field:
        inventor_count = df[inventor_field].astype(str).str.split(';|；').str.len()
        features_list.append(inventor_count.values.reshape(-1, 1))
        feature_names.append('发明人数量')
    else:
        features_list.append(np.ones((len(df), 1)))
        feature_names.append('发明人数量')
    
    # 5. 同族专利数量
    # 简单同族
    simple_family_field = None
    for field in ['简单同族数量', '简单同族个数', '简单同族数']:
        if field in df.columns:
            simple_family_field = field
            break
    
    if simple_family_field:
        simple_family = pd.to_numeric(df[simple_family_field], errors='coerce').fillna(1)
    else:
        simple_family = pd.Series([1] * len(df))
    features_list.append(simple_family.values.reshape(-1, 1))
    feature_names.append('简单同族个数')
    
    # 扩展同族
    extended_family_field = None
    for field in ['扩展同族数量', '扩展同族个数', '扩展同族数']:
        if field in df.columns:
            extended_family_field = field
            break
    
    if extended_family_field:
        extended_family = pd.to_numeric(df[extended_family_field], errors='coerce').fillna(1)
    else:
        extended_family = pd.Series([1] * len(df))
    features_list.append(extended_family.values.reshape(-1, 1))
    feature_names.append('扩展同族个数')
    
    # DocDB同族
    docdb_family_field = None
    for field in ['DocDB同族数量', 'DocDB同族个数', 'INPADOC同族数量', 'INPADOC同族个数']:
        if field in df.columns:
            docdb_family_field = field
            break
    
    if docdb_family_field:
        docdb_family = pd.to_numeric(df[docdb_family_field], errors='coerce').fillna(1)
    else:
        docdb_family = pd.Series([1] * len(df))
    features_list.append(docdb_family.values.reshape(-1, 1))
    feature_names.append('DocDB同族个数')
    
    # 组合所有特征
    features = np.hstack(features_list)
    
    print(f"  - 特征数量: {len(feature_names)}")
    print(f"  - 特征名称: {feature_names}")
    print(f"  - 特征维度: {features.shape}")
    
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 检查是否有NaN
    if np.any(np.isnan(features_scaled)):
        print("  警告：标准化后存在NaN值，进行填充...")
        features_scaled = np.nan_to_num(features_scaled, 0)
    
    return features_scaled, scaler, feature_names

def prepare_patent_data():
    """主函数：准备专利数据"""
    print("="*80)
    print("专利数据预处理")
    print("="*80)
    
    # 1. 读取内连接数据
    print("\n1. 读取内连接数据...")
    input_file = 'merged_patents_inner.xlsx'
    
    try:
        df = pd.read_excel(input_file)
        print(f"  成功读取 {len(df):,} 条记录")
        print(f"  字段数量: {len(df.columns)}")
    except FileNotFoundError:
        print(f"  错误：找不到文件 {input_file}")
        print("  请确保已运行合并脚本生成内连接数据")
        return
    
    # 2. 计算质量评分
    print("\n2. 计算时间标准化质量评分...")
    quality_scores = calculate_time_normalized_quality_score(df)
    df['质量评分'] = quality_scores
    
    # 3. 生成均衡标签
    print("\n3. 生成质量标签...")
    quality_labels = generate_balanced_labels(quality_scores)
    df['质量标签'] = quality_labels
    
    # 4. 准备文本数据
    print("\n4. 拼接文本字段...")
    texts = []
    text_lengths = []
    
    for idx, row in df.iterrows():
        if idx % 5000 == 0:
            print(f"  处理进度: {idx}/{len(df)}")
        
        combined_text = concatenate_all_text_fields(row)
        texts.append(combined_text)
        text_lengths.append(len(combined_text))
    
    df['组合文本'] = texts
    
    print(f"  文本长度统计:")
    print(f"    - 最小: {min(text_lengths):,} 字符")
    print(f"    - 最大: {max(text_lengths):,} 字符")
    print(f"    - 平均: {np.mean(text_lengths):,.0f} 字符")
    print(f"    - 中位数: {np.median(text_lengths):,.0f} 字符")
    
    # 5. 准备数值特征
    print("\n5. 准备数值特征...")
    numerical_features, scaler, feature_names = prepare_numerical_features_no_citations(df)
    
    # 将数值特征添加到DataFrame
    for i, name in enumerate(feature_names):
        df[f'数值特征_{name}'] = numerical_features[:, i]
    
    # 6. 选择必要的列准备输出
    print("\n6. 准备输出数据...")
    
    # 保留的列
    output_columns = ['标准化公开号', '标题 (中文)', '组合文本', '质量标签', '质量评分']
    
    # 添加数值特征列
    for name in feature_names:
        output_columns.append(f'数值特征_{name}')
    
    # 添加其他可能有用的元数据
    metadata_columns = ['申请人', '申请日', '公开（公告）日', 'IPC主分类号', '专利类型']
    for col in metadata_columns:
        if col in df.columns:
            output_columns.append(col)
    
    # 创建输出DataFrame
    output_df = df[output_columns].copy()
    
    # 7. 划分训练集和测试集
    print("\n7. 划分数据集...")
    
    # 确保每个类别在训练集和测试集中都有
    train_df, test_df = train_test_split(
        output_df, 
        test_size=0.2, 
        random_state=42,
        stratify=output_df['质量标签']  # 分层采样
    )
    
    print(f"  训练集: {len(train_df):,} 条")
    print(f"  测试集: {len(test_df):,} 条")
    
    # 验证训练集标签分布
    print("\n训练集标签分布:")
    train_label_counts = train_df['质量标签'].value_counts()
    for label in ['A', 'B', 'C']:
        count = train_label_counts.get(label, 0)
        percentage = count / len(train_df) * 100
        print(f"  {label}级: {count:,}条 ({percentage:.1f}%)")
    
    # 验证测试集标签分布
    print("\n测试集标签分布:")
    test_label_counts = test_df['质量标签'].value_counts()
    for label in ['A', 'B', 'C']:
        count = test_label_counts.get(label, 0)
        percentage = count / len(test_df) * 100
        print(f"  {label}级: {count:,}条 ({percentage:.1f}%)")
    
    # 8. 保存处理后的数据
    print("\n8. 保存数据...")
    
    # 保存训练集
    train_df.to_excel('train_data_prepared.xlsx', index=False)
    print(f"  ✓ 训练集已保存: train_data_prepared.xlsx")
    
    # 保存测试集
    test_df.to_excel('test_data_prepared.xlsx', index=False)
    print(f"  ✓ 测试集已保存: test_data_prepared.xlsx")
    
    # 保存完整数据（包含所有信息）
    output_df.to_excel('all_data_prepared.xlsx', index=False)
    print(f"  ✓ 完整数据已保存: all_data_prepared.xlsx")
    
    # 保存scaler
    import joblib
    joblib.dump(scaler, 'numerical_scaler.pkl')
    print(f"  ✓ 数值特征标准化器已保存: numerical_scaler.pkl")
    
    # 保存特征名称
    with open('feature_names.txt', 'w', encoding='utf-8') as f:
        for name in feature_names:
            f.write(name + '\n')
    print(f"  ✓ 特征名称已保存: feature_names.txt")
    
    print("\n" + "="*80)
    print("数据预处理完成！")
    print("="*80)
    
    print("\n下一步：")
    print("1. 运行修改后的模型训练脚本")
    print("2. 使用 train_data_prepared.xlsx 和 test_data_prepared.xlsx")
    print("3. 模型将自动处理长文本（2048 tokens）和数值特征")
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = prepare_patent_data()
