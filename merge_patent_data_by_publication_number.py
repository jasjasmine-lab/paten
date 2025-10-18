#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过公开号合并两份专利数据
将专利数据1.XLSX和专利数据2.xlsx通过公开号字段进行连接
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def analyze_publication_numbers():
    """分析两个文件中的公开号字段"""
    print("="*60)
    print("分析公开号字段")
    print("="*60)
    
    # 读取两个文件
    print("\n1. 读取数据文件...")
    df1 = pd.read_excel('专利数据1.XLSX')
    df2 = pd.read_excel('专利数据2.xlsx')
    
    print(f"   专利数据1.XLSX: {df1.shape[0]} 行 × {df1.shape[1]} 列")
    print(f"   专利数据2.xlsx: {df2.shape[0]} 行 × {df2.shape[1]} 列")
    
    # 查找公开号字段
    print("\n2. 公开号字段信息:")
    
    # 数据1的公开号字段
    pub_col1 = None
    for col in df1.columns:
        if '公开' in col and ('号' in col or '公告' in col):
            pub_col1 = col
            print(f"\n   专利数据1的公开号字段: {col}")
            print(f"     - 非空值: {df1[col].notna().sum()}/{len(df1)} ({df1[col].notna().sum()/len(df1)*100:.1f}%)")
            print(f"     - 唯一值: {df1[col].nunique()}")
            print(f"     - 示例: {df1[col].iloc[0]}")
            break
    
    # 数据2的公开号字段
    pub_col2 = None
    for col in df2.columns:
        if '公开' in col and ('号' in col or '公告' in col):
            pub_col2 = col
            print(f"\n   专利数据2的公开号字段: {col}")
            print(f"     - 非空值: {df2[col].notna().sum()}/{len(df2)} ({df2[col].notna().sum()/len(df2)*100:.1f}%)")
            print(f"     - 唯一值: {df2[col].nunique()}")
            print(f"     - 示例: {df2[col].iloc[0]}")
            break
    
    return df1, df2, pub_col1, pub_col2

def standardize_publication_number(pub_num):
    """标准化公开号格式"""
    if pd.isna(pub_num):
        return None
    
    # 转换为字符串并去除空格
    pub_num = str(pub_num).strip()
    
    # 去除可能的版本号后缀（如 A、B、C等）
    # 有些公开号可能有版本标记
    # 例如：CN101234567A -> CN101234567
    import re
    # 保留基本的公开号格式
    pub_num = re.sub(r'[A-Z]+$', '', pub_num)
    
    return pub_num

def merge_patent_data():
    """合并两份专利数据"""
    print("\n" + "="*60)
    print("合并专利数据")
    print("="*60)
    
    # 获取数据和公开号字段
    df1, df2, pub_col1, pub_col2 = analyze_publication_numbers()
    
    if not pub_col1 or not pub_col2:
        print("\n✗ 错误：无法找到公开号字段")
        return None
    
    # 标准化公开号
    print("\n3. 标准化公开号...")
    df1['标准化公开号'] = df1[pub_col1].apply(standardize_publication_number)
    df2['标准化公开号'] = df2[pub_col2].apply(standardize_publication_number)
    
    # 分析公开号匹配情况
    print("\n4. 分析公开号匹配情况...")
    
    # 找出共同的公开号
    common_pub_nums = set(df1['标准化公开号'].dropna()) & set(df2['标准化公开号'].dropna())
    
    print(f"   数据1中的唯一公开号: {df1['标准化公开号'].nunique()}")
    print(f"   数据2中的唯一公开号: {df2['标准化公开号'].nunique()}")
    print(f"   共同的公开号: {len(common_pub_nums)}")
    
    # 为避免列名冲突，给数据2的列添加后缀
    print("\n5. 准备合并...")
    
    # 保留原始的公开号列
    df1_for_merge = df1.copy()
    df2_for_merge = df2.copy()
    
    # 重命名数据2的列（除了标准化公开号）
    df2_columns_renamed = {}
    for col in df2_for_merge.columns:
        if col != '标准化公开号':
            # 如果列名在df1中存在，添加后缀
            if col in df1_for_merge.columns:
                df2_columns_renamed[col] = col + '_data2'
            else:
                df2_columns_renamed[col] = col
    
    df2_for_merge.rename(columns=df2_columns_renamed, inplace=True)
    
    # 执行合并
    print("\n6. 执行数据合并...")
    
    # 使用inner join（只保留匹配的记录）
    merged_inner = pd.merge(
        df1_for_merge,
        df2_for_merge,
        on='标准化公开号',
        how='inner'
    )
    
    # 使用left join（保留所有数据1的记录）
    merged_left = pd.merge(
        df1_for_merge,
        df2_for_merge,
        on='标准化公开号',
        how='left'
    )
    
    # 使用outer join（保留所有记录）
    merged_outer = pd.merge(
        df1_for_merge,
        df2_for_merge,
        on='标准化公开号',
        how='outer'
    )
    
    print(f"\n合并结果:")
    print(f"   内连接（只保留匹配）: {merged_inner.shape[0]} 行 × {merged_inner.shape[1]} 列")
    print(f"   左连接（保留数据1）: {merged_left.shape[0]} 行 × {merged_left.shape[1]} 列")
    print(f"   全连接（保留所有）: {merged_outer.shape[0]} 行 × {merged_outer.shape[1]} 列")
    
    # 保存合并结果
    print("\n7. 保存合并结果...")
    
    # 保存内连接结果（只包含匹配的记录）
    merged_inner.to_excel('merged_patents_inner.xlsx', index=False)
    print(f"   ✓ 内连接结果已保存到: merged_patents_inner.xlsx")
    
    # 保存左连接结果（包含所有数据1的记录）
    merged_left.to_excel('merged_patents_left.xlsx', index=False)
    print(f"   ✓ 左连接结果已保存到: merged_patents_left.xlsx")
    
    # 保存全连接结果（包含所有记录）
    merged_outer.to_excel('merged_patents_outer.xlsx', index=False)
    print(f"   ✓ 全连接结果已保存到: merged_patents_outer.xlsx")
    
    return merged_inner, merged_left, merged_outer

def analyze_merged_data(merged_df, merge_type):
    """分析合并后的数据"""
    print(f"\n{'='*60}")
    print(f"{merge_type}合并结果分析")
    print('='*60)
    
    # 关键字段完整性分析
    print("\n关键字段完整性:")
    
    # 来自数据1的字段
    data1_fields = ['技术功效', '技术手段', '权利要求数']
    for field in data1_fields:
        if field in merged_df.columns:
            non_null = merged_df[field].notna().sum()
            pct = non_null / len(merged_df) * 100
            print(f"  {field}: {non_null}/{len(merged_df)} ({pct:.1f}%)")
    
    # 来自数据2的字段
    data2_fields = ['标题 (中文)', '摘要 (中文)', '首项权利要求', '技术功效句', '用途', '家族被引证次数']
    for field in data2_fields:
        # 检查原始字段名或带后缀的字段名
        if field in merged_df.columns:
            col_name = field
        elif field + '_data2' in merged_df.columns:
            col_name = field + '_data2'
        else:
            continue
        
        non_null = merged_df[col_name].notna().sum()
        pct = non_null / len(merged_df) * 100
        print(f"  {col_name}: {non_null}/{len(merged_df)} ({pct:.1f}%)")
    
    # 数据来源分析
    print(f"\n数据来源分析:")
    only_data1 = merged_df['标题 (中文)'].isna().sum() if '标题 (中文)' in merged_df.columns else 0
    only_data2 = merged_df['技术功效'].isna().sum() if '技术功效' in merged_df.columns else 0
    both = len(merged_df) - only_data1 - only_data2
    
    if merge_type == "全连接":
        print(f"  只在数据1中: {only_data2} 条")
        print(f"  只在数据2中: {only_data1} 条")
        print(f"  两个数据集都有: {both} 条")

def main():
    """主函数"""
    print("通过公开号连接两份专利数据")
    print("="*60)
    
    try:
        # 执行合并
        merged_inner, merged_left, merged_outer = merge_patent_data()
        
        if merged_inner is not None:
            # 分析合并结果
            print("\n" + "="*60)
            print("合并结果详细分析")
            print("="*60)
            
            analyze_merged_data(merged_inner, "内连接")
            analyze_merged_data(merged_outer, "全连接")
            
            print("\n" + "="*60)
            print("合并完成！")
            print("="*60)
            print("\n生成的文件说明:")
            print("  1. merged_patents_inner.xlsx - 只包含两个数据集都有的专利")
            print("  2. merged_patents_left.xlsx - 包含所有数据1的专利，匹配数据2的信息")
            print("  3. merged_patents_outer.xlsx - 包含两个数据集的所有专利")
            print("\n建议:")
            print("  - 如果需要完整的数据集，使用 merged_patents_outer.xlsx")
            print("  - 如果只需要有完整信息的专利，使用 merged_patents_inner.xlsx")
            print("  - 如果以数据1为主，补充数据2的信息，使用 merged_patents_left.xlsx")
            
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
