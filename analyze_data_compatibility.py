#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专利数据文件适配性分析脚本
分析专利数据1.XLSX和专利数据2.xlsx是否与当前模型兼容
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def analyze_file(file_path, file_name):
    """分析单个文件的数据结构和质量"""
    print(f"\n{'='*60}")
    print(f"分析文件: {file_name}")
    print('='*60)
    
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        print(f"✓ 成功读取文件")
        print(f"  数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        
        # 1. 列名分析
        print(f"\n1. 文件包含的列 ({len(df.columns)} 个):")
        for i, col in enumerate(df.columns, 1):
            # 检查数据类型
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            null_pct = (df[col].isna().sum() / len(df)) * 100
            print(f"   {i:2d}. {col}")
            print(f"       类型: {dtype}, 非空值: {non_null}/{len(df)} ({100-null_pct:.1f}%)")
        
        # 2. 检查必需的文本字段
        print("\n2. 模型必需的文本字段检查:")
        text_fields = {
            '标题 (中文)': ['标题', 'title', '名称', '专利名称'],
            '摘要 (中文)': ['摘要', 'abstract', '简介'],
            '首项权利要求': ['权利要求', '权利', 'claim', '首项权利'],
            '技术功效句': ['技术功效', '功效', '技术效果'],
            '用途': ['用途', '应用', '应用领域']
        }
        
        found_fields = {}
        missing_fields = []
        
        for required_field, possible_names in text_fields.items():
            found = False
            for col in df.columns:
                if required_field in col or any(name in col.lower() for name in possible_names):
                    found_fields[required_field] = col
                    found = True
                    print(f"   ✓ {required_field} -> 找到: {col}")
                    break
            if not found:
                missing_fields.append(required_field)
                print(f"   ✗ {required_field} -> 未找到")
        
        # 3. 检查质量评分计算所需字段
        print("\n3. 质量评分计算所需字段检查:")
        score_fields = {
            '家族被引证次数': ['被引', '引证', '引用', 'citation', '被引次数'],
            '预估到期日': ['到期', '失效', 'expire', '到期日'],
            '公开（公告）日': ['公开', '公告', '申请日', 'publication', '公布日']
        }
        
        found_score_fields = {}
        missing_score_fields = []
        
        for required_field, possible_names in score_fields.items():
            found = False
            for col in df.columns:
                if required_field in col or any(name in col.lower() for name in possible_names):
                    found_score_fields[required_field] = col
                    found = True
                    print(f"   ✓ {required_field} -> 找到: {col}")
                    # 检查数据类型
                    if '日' in required_field or '日期' in col:
                        # 尝试转换为日期
                        try:
                            pd.to_datetime(df[col], errors='coerce')
                            print(f"       日期格式: 可转换")
                        except:
                            print(f"       日期格式: 需要处理")
                    break
            if not found:
                missing_score_fields.append(required_field)
                print(f"   ✗ {required_field} -> 未找到")
        
        # 4. 检查是否已有质量标签
        print("\n4. 质量标签检查:")
        label_found = False
        for col in df.columns:
            if '质量' in col and '标签' in col:
                label_found = True
                print(f"   ✓ 找到现有质量标签列: {col}")
                print(f"     标签分布: {df[col].value_counts().to_dict()}")
                break
        
        if not label_found:
            print("   ✗ 未找到质量标签列（需要生成）")
        
        # 5. 数据质量统计
        print("\n5. 数据质量概览:")
        print(f"   总行数: {len(df)}")
        print(f"   完全非空的行数: {df.dropna().shape[0]}")
        print(f"   包含空值的列数: {df.isna().any().sum()}/{len(df.columns)}")
        
        # 返回分析结果
        return {
            'success': True,
            'shape': df.shape,
            'columns': list(df.columns),
            'found_text_fields': found_fields,
            'missing_text_fields': missing_fields,
            'found_score_fields': found_score_fields,
            'missing_score_fields': missing_score_fields,
            'has_label': label_found,
            'df': df  # 返回数据框以供进一步分析
        }
        
    except Exception as e:
        print(f"✗ 读取文件失败: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def generate_compatibility_report(results1, results2):
    """生成适配性分析报告"""
    print(f"\n{'='*60}")
    print("适配性分析报告")
    print('='*60)
    
    files_info = [
        ('专利数据1.XLSX', results1),
        ('专利数据2.xlsx', results2)
    ]
    
    for file_name, result in files_info:
        print(f"\n【{file_name}】")
        
        if not result['success']:
            print(f"  ✗ 文件读取失败: {result['error']}")
            continue
        
        # 判断是否可以直接使用
        can_use_directly = (
            len(result['missing_text_fields']) == 0 and 
            result['has_label']
        )
        
        # 判断是否可以通过预处理使用
        can_preprocess = (
            len(result['missing_text_fields']) <= 2 and  # 允许缺少最多2个文本字段
            len(result['missing_score_fields']) == 0  # 必须有所有评分字段
        )
        
        print(f"\n  适配性评估:")
        if can_use_directly:
            print("  ✓ 可以直接用于模型训练/预测")
        elif can_preprocess:
            print("  ⚠ 需要预处理后才能使用")
        else:
            print("  ✗ 缺少关键字段，不适合当前模型")
        
        print(f"\n  详细分析:")
        print(f"    - 文本字段: 找到 {len(result['found_text_fields'])}/5")
        print(f"    - 评分字段: 找到 {len(result['found_score_fields'])}/3")
        print(f"    - 质量标签: {'有' if result['has_label'] else '无（需生成）'}")
        
        if result['missing_text_fields']:
            print(f"\n  缺失的文本字段:")
            for field in result['missing_text_fields']:
                print(f"    - {field}")
        
        if result['missing_score_fields']:
            print(f"\n  缺失的评分字段:")
            for field in result['missing_score_fields']:
                print(f"    - {field}")
        
        # 预处理建议
        if not can_use_directly and can_preprocess:
            print(f"\n  预处理建议:")
            if not result['has_label']:
                print("    1. 使用评分字段计算质量标签")
            if result['missing_text_fields']:
                print(f"    2. 处理缺失的文本字段（使用其他字段替代或填充）")

def main():
    """主函数"""
    print("="*60)
    print("专利数据文件与模型适配性分析")
    print("="*60)
    
    # 分析两个文件
    results1 = analyze_file('专利数据1.XLSX', '专利数据1.XLSX')
    results2 = analyze_file('专利数据2.xlsx', '专利数据2.xlsx')
    
    # 生成适配性报告
    generate_compatibility_report(results1, results2)
    
    # 保存详细报告
    print(f"\n{'='*60}")
    print("保存分析报告...")
    print('='*60)
    
    report = []
    report.append("# 专利数据适配性分析报告\n")
    report.append(f"分析时间: {pd.Timestamp.now()}\n")
    report.append("\n## 模型需求\n")
    report.append("### 文本字段（用于特征提取）:\n")
    report.append("- 标题 (中文)\n")
    report.append("- 摘要 (中文)\n")
    report.append("- 首项权利要求\n")
    report.append("- 技术功效句\n")
    report.append("- 用途\n")
    report.append("\n### 评分字段（用于标签生成）:\n")
    report.append("- 家族被引证次数\n")
    report.append("- 预估到期日\n")
    report.append("- 公开（公告）日\n")
    
    for file_name, result in [('专利数据1.XLSX', results1), ('专利数据2.xlsx', results2)]:
        report.append(f"\n## {file_name}\n")
        if result['success']:
            report.append(f"- 数据规模: {result['shape'][0]} 行 × {result['shape'][1]} 列\n")
            report.append(f"- 文本字段匹配: {len(result['found_text_fields'])}/5\n")
            report.append(f"- 评分字段匹配: {len(result['found_score_fields'])}/3\n")
            report.append(f"- 质量标签: {'已存在' if result['has_label'] else '需生成'}\n")
            
            # 适配性结论
            if len(result['missing_text_fields']) == 0 and result['has_label']:
                report.append("- **结论**: ✅ 可直接使用\n")
            elif len(result['missing_text_fields']) <= 2 and len(result['missing_score_fields']) == 0:
                report.append("- **结论**: ⚠️ 需要预处理\n")
            else:
                report.append("- **结论**: ❌ 不适配\n")
        else:
            report.append(f"- 错误: {result['error']}\n")
    
    # 保存报告
    with open('data_compatibility_report.md', 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print("✓ 报告已保存到: data_compatibility_report.md")
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)

if __name__ == "__main__":
    main()
