#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进的V3模型是否能正常运行
"""

# 设置离线模式，使用本地缓存
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import sys
import traceback

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def test_model():
    """测试模型是否能正常初始化和前向传播"""
    try:
        print("\n" + "="*60)
        print("测试模型V3 (ResNet增强版)")
        print("="*60)
        
        # 导入模型
        from chunked_roberta_v3_resnet import ChunkedRoBERTaBiLSTMCrossAttentionWithResNet
        
        # 加载分词器
        print("\n1. 加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        
        # 初始化模型
        print("\n2. 初始化模型...")
        model = ChunkedRoBERTaBiLSTMCrossAttentionWithResNet(
            num_classes=3,
            lstm_hidden_size=384,  # 增大的隐藏层
            lstm_layers=2,
            numerical_dim=7,
            fusion_hidden_dim=384,  # 增大的融合层
            dropout_rate=0.35,
            aggregation='attention',
            use_resnet=True,
            resnet_text_blocks=3,
            resnet_fusion_blocks=2
        ).to(device)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n模型参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  冻结参数: {total_params - trainable_params:,}")
        
        # 创建测试输入
        print("\n3. 创建测试输入...")
        batch_size = 2
        max_chunks = 16  # 增加块数
        chunk_size = 512  # RoBERTa最大支持512
        numerical_dim = 7
        
        # 模拟输入
        test_text = "这是一个测试专利的摘要内容。" * 50
        
        # 创建块
        input_ids_list = []
        attention_mask_list = []
        
        for i in range(max_chunks):
            if i < 3:  # 只填充前3个块
                encoding = tokenizer.encode_plus(
                    test_text,
                    truncation=True,
                    padding='max_length',
                    max_length=chunk_size,
                    return_tensors='pt'
                )
                input_ids_list.append(encoding['input_ids'])
                attention_mask_list.append(encoding['attention_mask'])
            else:
                # 空块
                input_ids_list.append(torch.zeros(1, chunk_size, dtype=torch.long))
                attention_mask_list.append(torch.zeros(1, chunk_size, dtype=torch.long))
        
        # Stack并复制到batch
        input_ids = torch.cat(input_ids_list, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
        attention_mask = torch.cat(attention_mask_list, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
        numerical_features = torch.randn(batch_size, numerical_dim)
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        numerical_features = numerical_features.to(device)
        
        print(f"\n输入形状:")
        print(f"  input_ids: {input_ids.shape}")
        print(f"  attention_mask: {attention_mask.shape}")
        print(f"  numerical_features: {numerical_features.shape}")
        
        # 前向传播测试
        print("\n4. 测试前向传播...")
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, numerical_features)
        
        print(f"\n输出形状: {outputs.shape}")
        print(f"输出示例: {outputs[0].cpu().numpy()}")
        
        # 测试损失计算
        print("\n5. 测试损失计算...")
        labels = torch.tensor([0, 1], dtype=torch.long).to(device)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        print(f"损失值: {loss.item():.4f}")
        
        # 测试梯度反向传播
        print("\n6. 测试梯度反向传播...")
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask, numerical_features)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 检查梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                if grad_norm > 0:
                    print(f"  {name}: 梯度范数 = {grad_norm:.6f}")
                    break
        
        if has_grad:
            print("\n✓ 梯度反向传播正常")
        else:
            print("\n✗ 梯度反向传播失败")
        
        # 测试优化器步进
        print("\n7. 测试优化器更新...")
        optimizer.step()
        print("✓ 优化器更新成功")
        
        print("\n" + "="*60)
        print("✓ 所有测试通过！模型可以正常使用")
        print("="*60)
        
        # 显存使用情况
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\nGPU显存使用:")
            print(f"  已分配: {allocated:.2f} GB")
            print(f"  已预留: {reserved:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        print("\n详细错误信息:")
        traceback.print_exc()
        return False

def test_resnet_components():
    """单独测试ResNet组件"""
    try:
        print("\n" + "="*60)
        print("测试ResNet组件")
        print("="*60)
        
        from chunked_roberta_v3_resnet import ResidualBlock, TextResNet, FusionResNet
        
        # 测试ResidualBlock
        print("\n1. 测试ResidualBlock...")
        block = ResidualBlock(hidden_dim=256, dropout_rate=0.1)
        x = torch.randn(2, 256)
        out = block(x)
        assert out.shape == x.shape
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {out.shape}")
        print("  ✓ ResidualBlock测试通过")
        
        # 测试TextResNet
        print("\n2. 测试TextResNet...")
        text_resnet = TextResNet(hidden_dim=768, num_blocks=3, dropout_rate=0.1)
        x = torch.randn(2, 768)
        out = text_resnet(x)
        assert out.shape == x.shape
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {out.shape}")
        print("  ✓ TextResNet测试通过")
        
        # 测试FusionResNet
        print("\n3. 测试FusionResNet...")
        fusion_resnet = FusionResNet(fusion_dim=384, num_blocks=2, dropout_rate=0.2)
        x = torch.randn(2, 384)
        out = fusion_resnet(x)
        assert out.shape == x.shape
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {out.shape}")
        print("  ✓ FusionResNet测试通过")
        
        print("\n✓ 所有ResNet组件测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ ResNet组件测试失败: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 测试ResNet组件
    resnet_ok = test_resnet_components()
    
    # 测试完整模型
    model_ok = test_model()
    
    if resnet_ok and model_ok:
        print("\n" + "="*60)
        print("🎉 所有测试通过！可以开始训练")
        print("运行命令: python chunked_roberta_v3_resnet.py")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️ 部分测试失败，请检查错误信息")
        print("="*60)
