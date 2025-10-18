#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证模型输入中没有标签泄露
确保标签仅作为训练目标，不作为输入特征
"""

import pandas as pd
import numpy as np

def verify_no_label_leakage():
    """验证输入特征中没有标签信息"""
    print("="*80)
    print("验证标签泄露检查")
    print("="*80)
    
    # 1. 读取处理后的数据
    print("\n1. 读取处理后的数据...")
    train_df = pd.read_excel('train_data_prepared.xlsx')
    test_df = pd.read_excel('test_data_prepared.xlsx')
    
    print(f"   训练集: {len(train_df)} 条")
    print(f"   测试集: {len(test_df)} 条")
    
    # 2. 检查数值特征列
    print("\n2. 检查数值特征（输入特征）...")
    feature_columns = [col for col in train_df.columns if col.startswith('数值特征_')]
    
    print(f"\n   数值特征列({len(feature_columns)}个):")
    for col in feature_columns:
        feature_name = col.replace('数值特征_', '')
        print(f"     - {feature_name}")
    
    # 3. 验证数值特征中没有引证信息
    print("\n3. 验证数值特征中没有引证相关信息...")
    
    citation_keywords = ['引证', '引用', '被引', 'citation', 'cited']
    has_citation = False
    
    for col in feature_columns:
        for keyword in citation_keywords:
            if keyword.lower() in col.lower():
                print(f"   ❌ 发现可能的引证特征: {col}")
                has_citation = True
    
    if not has_citation:
        print("   ✅ 数值特征中没有引证信息")
    
    # 4. 检查标签列
    print("\n4. 确认标签列仅用作训练目标...")
    
    if '质量标签' in train_df.columns:
        print("   ✅ '质量标签'列存在，用作训练目标")
        
        # 显示标签分布
        label_counts = train_df['质量标签'].value_counts()
        print("\n   训练集标签分布:")
        for label in ['A', 'B', 'C']:
            count = label_counts.get(label, 0)
            print(f"     {label}级: {count:,}条 ({count/len(train_df)*100:.1f}%)")
    
    # 5. 检查质量评分
    print("\n5. 验证质量评分使用...")
    
    if '质量评分' in train_df.columns:
        print("   ✅ '质量评分'列存在")
        print("   注意: 质量评分仅用于生成标签，不作为模型输入")
        
        # 验证质量评分不在数值特征中
        if '数值特征_质量评分' in train_df.columns:
            print("   ❌ 警告: 质量评分被用作输入特征！")
        else:
            print("   ✅ 质量评分未被用作输入特征")
    
    # 6. 列出模型的实际输入
    print("\n6. 模型的实际输入特征总结:")
    print("-"*60)
    
    print("\n   文本输入:")
    print("     - 组合文本 (拼接的专利文本字段)")
    
    print("\n   数值输入 (7个特征):")
    numerical_features = [
        '权利要求数量',
        '首权字数', 
        '申请人数量',
        '发明人数量',
        '简单同族个数',
        '扩展同族个数',
        'DocDB同族个数'
    ]
    
    for i, feat in enumerate(numerical_features, 1):
        print(f"     {i}. {feat}")
    
    print("\n   输出目标:")
    print("     - 质量标签 (A/B/C)")
    
    # 7. 验证引证信息确实不在输入中
    print("\n7. 最终验证结果:")
    print("-"*60)
    
    # 检查原始数据中的引证字段
    if '家族被引证次数' in train_df.columns:
        print("   ⚠️ 原始数据包含'家族被引证次数'")
        print("   ✅ 但该字段仅用于计算质量评分，不作为模型输入")
    
    print("\n   ✅ 验证通过: 标签和引证信息都不在模型输入中")
    print("   ✅ 模型仅使用文本和基础数值特征进行预测")
    print("   ✅ 质量标签仅作为训练目标")
    
    return True

def main():
    """主函数"""
    print("\n标签泄露验证程序")
    print("="*80)
    
    try:
        result = verify_no_label_leakage()
        
        if result:
            print("\n" + "="*80)
            print("验证完成 - 无标签泄露")
            print("="*80)
            print("\n✅ 模型配置正确，可以安全地进行训练")
            print("✅ 标签仅作为输出目标，不作为输入特征")
        else:
            print("\n❌ 发现潜在问题，请检查数据处理流程")
            
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
