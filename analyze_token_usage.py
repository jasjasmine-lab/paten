#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细分析文本的Token使用情况
使用实际的RoBERTa分词器进行精确计算
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_token_distribution():
    """分析token数量分布"""
    print("="*80)
    print("Token使用情况详细分析")
    print("="*80)
    
    # 1. 加载RoBERTa分词器
    print("\n1. 加载RoBERTa分词器...")
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    print(f"   分词器类型: {tokenizer.__class__.__name__}")
    print(f"   词汇表大小: {tokenizer.vocab_size}")
    
    # 2. 读取处理后的数据
    print("\n2. 读取处理后的数据...")
    all_df = pd.read_excel('all_data_prepared.xlsx')
    print(f"   总记录数: {len(all_df):,}")
    
    # 3. 计算每条记录的实际token数
    print("\n3. 计算实际token数量（使用进度条）...")
    token_counts = []
    text_lengths = []
    
    # 使用进度条
    for idx, text in tqdm(enumerate(all_df['组合文本']), total=len(all_df), desc="分词处理"):
        # 使用实际的分词器计算token数
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_count = len(tokens)
        token_counts.append(token_count)
        text_lengths.append(len(text))
    
    # 添加到DataFrame
    all_df['实际token数'] = token_counts
    all_df['字符数'] = text_lengths
    all_df['字符/token比'] = all_df['字符数'] / all_df['实际token数']
    
    # 4. 统计分析
    print("\n4. Token数量统计分析")
    print("-"*60)
    
    token_stats = pd.Series(token_counts)
    print(f"   最小值: {token_stats.min()} tokens")
    print(f"   25%分位: {token_stats.quantile(0.25):.0f} tokens")
    print(f"   中位数: {token_stats.median():.0f} tokens")
    print(f"   75%分位: {token_stats.quantile(0.75):.0f} tokens")
    print(f"   95%分位: {token_stats.quantile(0.95):.0f} tokens")
    print(f"   99%分位: {token_stats.quantile(0.99):.0f} tokens")
    print(f"   最大值: {token_stats.max()} tokens")
    print(f"   平均值: {token_stats.mean():.1f} tokens")
    print(f"   标准差: {token_stats.std():.1f} tokens")
    
    # 5. 检查超出限制的情况
    print("\n5. Token限制检查（2048 tokens）")
    print("-"*60)
    
    exceed_2048 = (token_stats > 2048).sum()
    exceed_1024 = (token_stats > 1024).sum()
    exceed_512 = (token_stats > 512).sum()
    
    print(f"   超过2048 tokens: {exceed_2048:,}条 ({exceed_2048/len(all_df)*100:.2f}%)")
    print(f"   超过1024 tokens: {exceed_1024:,}条 ({exceed_1024/len(all_df)*100:.2f}%)")
    print(f"   超过512 tokens: {exceed_512:,}条 ({exceed_512/len(all_df)*100:.2f}%)")
    
    # 6. 显示超出2048的样本
    if exceed_2048 > 0:
        print(f"\n6. 超出2048 tokens的样本详情")
        print("-"*60)
        
        exceed_df = all_df[all_df['实际token数'] > 2048].sort_values('实际token数', ascending=False)
        
        print(f"   共有 {len(exceed_df)} 条记录超出限制")
        print("\n   前10条超限记录:")
        
        for idx, row in exceed_df.head(10).iterrows():
            print(f"\n   记录 {idx}:")
            print(f"     公开号: {row['标准化公开号']}")
            print(f"     标题: {row.get('标题 (中文)', 'N/A')[:50]}...")
            print(f"     Token数: {row['实际token数']} (超出 {row['实际token数']-2048} tokens)")
            print(f"     字符数: {row['字符数']}")
            print(f"     质量标签: {row['质量标签']}")
    
    # 7. 字符与token的关系分析
    print("\n7. 中文字符与Token的转换关系")
    print("-"*60)
    
    ratio_stats = all_df['字符/token比'].describe()
    print(f"   平均比例: 1字符 ≈ {1/ratio_stats['mean']:.3f} tokens")
    print(f"   中位数比例: 1字符 ≈ {1/ratio_stats['50%']:.3f} tokens")
    print(f"   建议的字符限制: {2048 * ratio_stats['mean']:.0f} 字符（对应2048 tokens）")
    
    # 8. 推荐的处理策略
    print("\n8. 处理策略建议")
    print("-"*60)
    
    if exceed_2048 == 0:
        print("   ✅ 所有文本都在2048 tokens限制内，无需调整！")
    else:
        percentage = exceed_2048/len(all_df)*100
        
        if percentage < 1:
            print(f"   ⚠️ 只有{percentage:.2f}%的数据超限，影响很小")
            print("   建议策略：")
            print("     1. 保持2048 tokens限制")
            print("     2. 超限文本会自动截断，影响可忽略")
            
        elif percentage < 5:
            print(f"   ⚠️ 有{percentage:.2f}%的数据超限，需要注意")
            print("   建议策略：")
            print("     1. 可以保持2048 tokens，接受少量截断")
            print("     2. 或将字符限制降到12000以确保99%的数据完整")
            
        else:
            print(f"   ❌ 有{percentage:.2f}%的数据超限，需要调整")
            print("   建议策略：")
            print("     1. 增加MAX_LENGTH到3000或4096")
            print("     2. 或减少拼接的字段数量")
            print("     3. 或限制每个字段的最大长度")
    
    # 9. 保存分析结果
    print("\n9. 保存分析结果...")
    
    # 保存详细数据
    analysis_df = all_df[['标准化公开号', '质量标签', '实际token数', '字符数', '字符/token比']].copy()
    analysis_df.to_excel('token_analysis_results.xlsx', index=False)
    print("   ✓ 详细结果已保存到: token_analysis_results.xlsx")
    
    # 生成分布图
    try:
        plt.figure(figsize=(12, 4))
        
        # Token分布直方图
        plt.subplot(1, 3, 1)
        plt.hist(token_counts, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=2048, color='r', linestyle='--', label='2048 limit')
        plt.xlabel('Token Count')
        plt.ylabel('Frequency')
        plt.title('Token Distribution')
        plt.legend()
        
        # 累积分布图
        plt.subplot(1, 3, 2)
        sorted_tokens = np.sort(token_counts)
        cumulative = np.arange(1, len(sorted_tokens) + 1) / len(sorted_tokens) * 100
        plt.plot(sorted_tokens, cumulative)
        plt.axvline(x=2048, color='r', linestyle='--', label='2048 limit')
        plt.axhline(y=95, color='g', linestyle='--', alpha=0.5, label='95%')
        plt.axhline(y=99, color='g', linestyle='--', alpha=0.5, label='99%')
        plt.xlabel('Token Count')
        plt.ylabel('Cumulative Percentage (%)')
        plt.title('Cumulative Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 字符vs Token散点图
        plt.subplot(1, 3, 3)
        plt.scatter(text_lengths, token_counts, alpha=0.3, s=1)
        plt.axhline(y=2048, color='r', linestyle='--', label='2048 token limit')
        plt.xlabel('Character Count')
        plt.ylabel('Token Count')
        plt.title('Characters vs Tokens')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('token_distribution.png', dpi=100)
        print("   ✓ 分布图已保存到: token_distribution.png")
        plt.close()
        
    except Exception as e:
        print(f"   ⚠️ 无法生成图表: {e}")
    
    return all_df, token_stats

def main():
    """主函数"""
    print("\n专利文本Token使用情况分析")
    print("使用实际的RoBERTa分词器进行精确计算")
    print("="*80)
    
    try:
        all_df, token_stats = analyze_token_distribution()
        
        # 最终结论
        print("\n" + "="*80)
        print("分析结论")
        print("="*80)
        
        exceed_count = (token_stats > 2048).sum()
        
        if exceed_count == 0:
            print("\n✅ 完美！所有文本都在2048 tokens限制内")
            print("   无需任何调整，可以直接开始训练")
        elif exceed_count < len(token_stats) * 0.01:
            print(f"\n✅ 良好！只有{exceed_count}条({exceed_count/len(token_stats)*100:.2f}%)超限")
            print("   超限数据会自动截断，影响极小")
            print("   建议保持当前配置，直接开始训练")
        else:
            print(f"\n⚠️ 注意！有{exceed_count}条({exceed_count/len(token_stats)*100:.2f}%)超限")
            print("   建议调整文本预处理策略或增加token限制")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
