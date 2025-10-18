#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析内连接后的数据集质量
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def analyze_data_quality():
    """分析合并后数据的质量"""
    
    print("="*80)
    print("内连接数据集质量分析")
    print("="*80)
    
    # 读取内连接结果
    print("\n1. 读取内连接数据...")
    df = pd.read_excel('merged_patents_inner.xlsx')
    print(f"   数据规模: {df.shape[0]} 行 × {df.shape[1]} 列")
    
    # 分析关键字段的完整性
    print("\n2. 关键字段完整性分析")
    print("-"*60)
    
    # 模型必需的文本字段
    text_fields = {
        '标题 (中文)': '专利标题',
        '摘要 (中文)': '专利摘要',
        '首项权利要求': '权利要求',
        '技术功效句': '技术功效描述',
        '用途': '应用领域',
        '技术功效': '技术功效（数据1）',
        '技术手段': '技术手段（数据1）'
    }
    
    print("\n文本字段完整性:")
    for field, desc in text_fields.items():
        if field in df.columns:
            non_null = df[field].notna().sum()
            null_count = df[field].isna().sum()
            pct = non_null / len(df) * 100
            avg_length = df[field].dropna().astype(str).str.len().mean()
            print(f"  {desc:20s}: {non_null:6d}/{len(df)} ({pct:5.1f}%) | 缺失: {null_count:5d} | 平均长度: {avg_length:.0f}字符")
    
    # 质量评分相关字段
    print("\n质量评分计算字段:")
    score_fields = {
        '家族被引证次数': '引用次数',
        '公开（公告）日': '公开日期',
        '预估到期日': '到期日期',
        '申请日': '申请日期'
    }
    
    for field, desc in score_fields.items():
        if field in df.columns:
            non_null = df[field].notna().sum()
            null_count = df[field].isna().sum()
            pct = non_null / len(df) * 100
            print(f"  {desc:20s}: {non_null:6d}/{len(df)} ({pct:5.1f}%) | 缺失: {null_count:5d}")
            
            # 对于数值型字段，显示统计信息
            if field == '家族被引证次数':
                valid_data = df[field].dropna()
                if len(valid_data) > 0:
                    print(f"    统计: 最小={valid_data.min():.0f}, 平均={valid_data.mean():.1f}, 中位数={valid_data.median():.0f}, 最大={valid_data.max():.0f}")
    
    # 分析数据分布
    print("\n3. 数据分布分析")
    print("-"*60)
    
    # 专利类型分布
    if '专利类型' in df.columns:
        print("\n专利类型分布:")
        type_counts = df['专利类型'].value_counts().head(5)
        for ptype, count in type_counts.items():
            pct = count / len(df) * 100
            print(f"  {ptype}: {count} ({pct:.1f}%)")
    
    # 申请年份分布
    if '申请日' in df.columns:
        df['申请年份'] = pd.to_datetime(df['申请日'], errors='coerce').dt.year
        year_counts = df['申请年份'].value_counts().sort_index().tail(5)
        print("\n最近5年申请分布:")
        for year, count in year_counts.items():
            if pd.notna(year):
                pct = count / len(df) * 100
                print(f"  {int(year)}年: {count} ({pct:.1f}%)")
    
    # 国家/地区分布
    if '申请人国家/地区' in df.columns:
        print("\n申请人国家/地区分布（前5）:")
        country_counts = df['申请人国家/地区'].value_counts().head(5)
        for country, count in country_counts.items():
            pct = count / len(df) * 100
            print(f"  {country}: {count} ({pct:.1f}%)")
    
    # IPC分类分布
    if 'IPC主分类' in df.columns or 'IPC主分类号' in df.columns:
        ipc_col = 'IPC主分类' if 'IPC主分类' in df.columns else 'IPC主分类号'
        print("\nIPC主分类分布（前5）:")
        # 提取IPC大类（前3个字符）
        df['IPC大类'] = df[ipc_col].astype(str).str[:3]
        ipc_counts = df['IPC大类'].value_counts().head(5)
        for ipc, count in ipc_counts.items():
            if ipc != 'nan':
                pct = count / len(df) * 100
                print(f"  {ipc}: {count} ({pct:.1f}%)")
    
    # 计算可用于质量评分的数据
    print("\n4. 质量评分可计算性分析")
    print("-"*60)
    
    required_fields = ['家族被引证次数', '公开（公告）日', '预估到期日']
    available_fields = [f for f in required_fields if f in df.columns]
    
    if len(available_fields) == len(required_fields):
        # 计算三个字段都非空的记录数
        complete_for_scoring = df[required_fields].notna().all(axis=1).sum()
        pct = complete_for_scoring / len(df) * 100
        print(f"可计算质量评分的记录: {complete_for_scoring}/{len(df)} ({pct:.1f}%)")
        
        # 分析缺失模式
        print("\n缺失模式分析:")
        missing_patterns = df[required_fields].isna().value_counts().head(5)
        for pattern, count in missing_patterns.items():
            pct = count / len(df) * 100
            if isinstance(pattern, tuple):
                pattern_str = ', '.join([f"{field}={'缺失' if miss else '有值'}" 
                                        for field, miss in zip(required_fields, pattern)])
            else:
                pattern_str = f"家族被引证次数={'缺失' if pattern else '有值'}"
            print(f"  {pattern_str}: {count} ({pct:.1f}%)")
    
    # 随机展示样本数据
    print("\n5. 随机样本数据展示")
    print("-"*60)
    
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    # 随机选择5条记录
    sample_indices = np.random.choice(df.index, min(5, len(df)), replace=False)
    
    # 选择要展示的关键字段
    display_fields = [
        '标准化公开号', '标题 (中文)', '摘要 (中文)', 
        '技术功效句', '用途', '家族被引证次数',
        '申请人', '申请日', '专利类型'
    ]
    
    # 过滤存在的字段
    display_fields = [f for f in display_fields if f in df.columns]
    
    for i, idx in enumerate(sample_indices, 1):
        print(f"\n样本 {i}:")
        print("="*60)
        record = df.loc[idx]
        
        for field in display_fields:
            value = record[field]
            if pd.notna(value):
                value_str = str(value)
                # 对长文本进行截断
                if field in ['摘要 (中文)', '技术功效句', '用途']:
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                print(f"  {field}: {value_str}")
            else:
                print(f"  {field}: [缺失]")
    
    # 数据质量总结
    print("\n" + "="*80)
    print("数据质量总结")
    print("="*80)
    
    # 计算整体完整性
    total_cells = len(df) * len(df.columns)
    non_null_cells = df.notna().sum().sum()
    overall_completeness = non_null_cells / total_cells * 100
    
    print(f"\n整体数据完整性: {overall_completeness:.1f}%")
    print(f"总记录数: {len(df):,}")
    print(f"总字段数: {len(df.columns)}")
    print(f"非空单元格: {non_null_cells:,} / {total_cells:,}")
    
    # 关键字段评估
    critical_fields = ['标题 (中文)', '摘要 (中文)', '首项权利要求', '家族被引证次数']
    critical_complete = sum([df[f].notna().sum() for f in critical_fields if f in df.columns])
    critical_total = len(df) * len([f for f in critical_fields if f in df.columns])
    critical_completeness = critical_complete / critical_total * 100 if critical_total > 0 else 0
    
    print(f"\n关键字段完整性: {critical_completeness:.1f}%")
    
    # 数据质量等级评估
    if overall_completeness >= 80 and critical_completeness >= 90:
        quality_grade = "优秀 ⭐⭐⭐⭐⭐"
    elif overall_completeness >= 70 and critical_completeness >= 80:
        quality_grade = "良好 ⭐⭐⭐⭐"
    elif overall_completeness >= 60 and critical_completeness >= 70:
        quality_grade = "中等 ⭐⭐⭐"
    else:
        quality_grade = "需改进 ⭐⭐"
    
    print(f"\n数据质量等级: {quality_grade}")
    
    # 建议
    print("\n改进建议:")
    if '技术功效句' in df.columns and df['技术功效句'].isna().sum() > len(df) * 0.1:
        print("  - 技术功效句字段有较多缺失，建议使用技术功效字段补充")
    if '用途' in df.columns and df['用途'].isna().sum() > len(df) * 0.1:
        print("  - 用途字段有较多缺失，可考虑从其他字段提取")
    if '家族被引证次数' in df.columns and df['家族被引证次数'].isna().sum() > len(df) * 0.2:
        print("  - 家族被引证次数缺失较多，质量评分计算时需要特殊处理")
    
    return df

if __name__ == "__main__":
    try:
        df = analyze_data_quality()
        print("\n分析完成！")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
