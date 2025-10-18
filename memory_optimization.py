#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化策略for超长序列训练
支持8192 tokens的高效训练
"""

import torch
import gc
from contextlib import contextmanager

class MemoryOptimizer:
    """内存优化工具类"""
    
    @staticmethod
    def clear_cache():
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    @staticmethod
    def get_memory_info():
        """获取当前GPU内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            return {
                'allocated': allocated,
                'reserved': reserved,
                'free': total - reserved,
                'total': total,
                'usage_percent': (reserved / total) * 100
            }
        return None
    
    @staticmethod
    @contextmanager
    def memory_efficient_mode():
        """内存高效模式上下文管理器"""
        # 进入时清理缓存
        MemoryOptimizer.clear_cache()
        
        # 设置cudnn为确定性模式，减少内存占用
        old_cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        
        try:
            yield
        finally:
            # 退出时恢复设置并清理
            torch.backends.cudnn.benchmark = old_cudnn_benchmark
            MemoryOptimizer.clear_cache()
    
    @staticmethod
    def estimate_batch_size(max_length, model_size='base'):
        """
        根据序列长度估算合适的批次大小
        
        Args:
            max_length: 最大序列长度
            model_size: 模型大小 ('base', 'large')
        
        Returns:
            推荐的批次大小
        """
        if not torch.cuda.is_available():
            return 1
        
        # 获取GPU内存
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # 基于经验的内存占用估算（GB）
        # RoBERTa-base + BiLSTM + CrossAttention
        if model_size == 'base':
            memory_per_sample = (max_length / 512) * 0.5  # 每512 tokens约0.5GB
        else:
            memory_per_sample = (max_length / 512) * 0.8  # large模型
        
        # 保留30%内存作为缓冲
        available_memory = total_memory * 0.7
        
        # 计算批次大小
        batch_size = max(1, int(available_memory / memory_per_sample))
        
        return batch_size
    
    @staticmethod
    def setup_mixed_precision():
        """
        设置混合精度训练
        返回GradScaler对象
        """
