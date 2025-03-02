import time
import torch
import numpy as np
from collections import defaultdict

class ModelMonitor:
    def __init__(self, print_per_forward=False):
        # 控制是否每次前向传播都打印信息的参数
        self.print_per_forward = print_per_forward
        
        # 全局级别监控
        self.global_stats = {
            "total_time": 0,
            "call_count": 0,
            "total_tokens": 0,
            "total_samples": 0
        }
        
        # 模块级别监控
        self.module_start_times = {}
        self.module_stats = defaultdict(lambda: {"total_time": 0, "call_count": 0})
        
        # 输入输出监控
        self.input_shapes = []
        self.output_shapes = []
    
    def register_hooks(self, model):
        """为模型和其子模块注册监控钩子"""
        # 为整个模型注册全局钩子
        model.register_forward_pre_hook(self.global_start_hook)
        model.register_forward_hook(self.global_end_hook)
        
        # 为所有子模块注册计时钩子
        for name, module in model.named_modules():
            if name:  # 跳过根模块
                module.register_forward_pre_hook(self.module_start_hook)
                module.register_forward_hook(self.module_end_hook)
    
    def global_start_hook(self, module, input):
        """全局开始计时钩子"""
        torch.cuda.synchronize()
        self.global_start_time = time.time()
        
        # 记录输入信息
        if input and len(input) > 0 and hasattr(input[0], 'shape'):
            input_ids = input[0]
            self.input_shapes.append(input_ids.shape)
            
            # 计算tokens
            batch_size, seq_len = input_ids.shape[:2]
            self.global_stats["total_tokens"] += batch_size * seq_len
            self.global_stats["total_samples"] += batch_size
            
            # 只在需要时打印
            if self.print_per_forward:
                print(f"输入shape: {input_ids.shape}, tokens: {batch_size * seq_len}")
    
    def global_end_hook(self, module, input, output):
        """全局结束计时钩子"""
        torch.cuda.synchronize()
        elapsed = (time.time() - self.global_start_time) * 1000
        
        self.global_stats["total_time"] += elapsed
        self.global_stats["call_count"] += 1
        
        # 记录输出信息
        if hasattr(output, 'logits') and hasattr(output.logits, 'shape'):
            self.output_shapes.append(output.logits.shape)
            if self.print_per_forward:
                print(f"输出logits形状: {output.logits.shape}")
        
        if self.print_per_forward:
            print(f"本次推理时间: {elapsed:.2f}毫秒")
        
        return output
    
    def module_start_hook(self, module, input):
        """模块级别开始计时钩子"""
        torch.cuda.synchronize()
        self.module_start_times[module] = time.time()
    
    def module_end_hook(self, module, input, output):
        """模块级别结束计时钩子"""
        if module not in self.module_start_times:
            return output
            
        torch.cuda.synchronize()
        elapsed = (time.time() - self.module_start_times[module]) * 1000
        
        module_name = f"{module.__class__.__name__}"
        self.module_stats[module_name]["total_time"] += elapsed
        self.module_stats[module_name]["call_count"] += 1
        
        return output
    
    def report(self):
        """生成监控报告"""
        print("\n========== 模型监控报告 ==========")
        
        # 全局统计信息
        print("\n----- 全局性能指标 -----")
        total_time = self.global_stats["total_time"]
        call_count = self.global_stats["call_count"]
        if call_count > 0:
            avg_time = total_time / call_count
            tokens_per_sec = self.global_stats["total_tokens"] / (total_time / 1000) if total_time > 0 else 0
            
            print(f"总推理次数: {call_count}")
            print(f"总推理时间: {total_time:.2f}毫秒 ({total_time/1000:.2f}秒)")
            print(f"平均推理时间: {avg_time:.2f}毫秒/次")
            print(f"处理样本总数: {self.global_stats['total_samples']}")
            print(f"处理token总数: {self.global_stats['total_tokens']}")
            print(f"吞吐量: {tokens_per_sec:.2f} tokens/秒")
        
        # 模块统计信息
        print("\n----- 模块性能指标 -----")
        sorted_modules = sorted(self.module_stats.items(), 
                              key=lambda x: x[1]["total_time"], 
                              reverse=True)
        
        for module_name, stats in sorted_modules[:20]:  # 只显示前20个耗时最长的模块
            avg_time = stats["total_time"] / stats["call_count"] if stats["call_count"] > 0 else 0
            time_percent = (stats["total_time"] / total_time * 100) if total_time > 0 else 0
            
            print(f"{module_name}: 总计 {stats['total_time']:.2f}ms, "
                  f"调用 {stats['call_count']}次, "
                  f"平均 {avg_time:.2f}ms/次, "
                  f"占比 {time_percent:.1f}%")
        
        # 输入输出统计
        print("\n----- 输入/输出统计 -----")
        print(f"输入形状示例: {self.input_shapes[-5:] if self.input_shapes else 'None'}")
        print(f"输出形状示例: {self.output_shapes[-5:] if self.output_shapes else 'None'}")
        
        print("\n===============================")
        
    def reset(self):
        """重置所有统计信息"""
        # 全局级别监控
        self.global_stats = {
            "total_time": 0,
            "call_count": 0,
            "total_tokens": 0,
            "total_samples": 0
        }
        
        # 模块级别监控
        self.module_start_times = {}
        self.module_stats = defaultdict(lambda: {"total_time": 0, "call_count": 0})
        
        # 输入输出监控
        self.input_shapes = []
        self.output_shapes = []
        
        print("已重置所有监控统计信息")

global_monitor = ModelMonitor(print_per_forward=False)