import time
import torch
import numpy as np
from collections import defaultdict

class ModelMonitor:
    def __init__(self, print_per_forward=False):
        # 基本设置
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
        
        # 样本处理时间
        self.sample_processing_times = []
        
        # 新增: prefill 和 decode 阶段监控 (保留数据收集但删除报告输出)
        self.prefill_stats = {
            "total_time": 0,
            "call_count": 0,
            "attention_time": 0,  # 注意力层耗时
            "sample_times": []
        }
        
        self.decode_stats = {
            "total_time": 0,
            "call_count": 0,
            "attention_time": 0,  # 注意力层耗时
            "token_times": [],
            "samples_total_time": []
        }
        
        # 当前样本状态跟踪
        self.current_sample_id = None
        self.current_phase = None  # "prefill" 或 "decode"
        self.current_sample_start_time = None
        self.current_sample_attention_time = 0
        self.current_sample_decode_times = []
        
        # 内部状态
        self._in_attention_layer = False
        self._attention_start_time = None
        self._max_out_len = 100  # 默认值，应当设置为实际max_out_len
        
        # 新增: 注意力层内部组件统计
        self.attention_component_stats = {
            "total_attention_time": 0,
            "components": defaultdict(lambda: {"total_time": 0, "call_count": 0})
        }
        self._current_attention_component = None
        self._component_start_time = None
        
        # 新增: 注意力层合并统计
        self.attention_layer_stats = {
            "total_time": 0,
            "call_count": 0,
            "layers": defaultdict(lambda: {"total_time": 0, "call_count": 0})
        }
        
        # 新增: 注意力计算统计
        self.attention_compute_stats = {
            "total_time": 0,
            "call_count": 0
        }
        self._in_attention_compute = False
        self._attention_compute_start_time = None
    
    def register_hooks(self, model):
        """为模型和其子模块注册监控钩子"""
        # 检查模型是否支持钩子
        if hasattr(model, 'register_forward_pre_hook'):
            # 为整个模型注册全局钩子
            model.register_forward_pre_hook(self.global_start_hook)
            model.register_forward_hook(self.global_end_hook)
    
        # 为所有子模块注册计时钩子
        for name, module in model.named_modules():
            if name and hasattr(module, 'register_forward_pre_hook'):  # 检查模块是否支持钩子
                # 为SparseLlamaAttention特殊处理
                if module.__class__.__name__ == "SparseLlamaAttention":
                    module.register_forward_pre_hook(self.attention_start_hook)
                    module.register_forward_hook(self.attention_end_hook)
                    
                    # 为SparseLlamaAttention内部组件注册钩子
                    for comp_name, comp_module in module.named_children():
                        if hasattr(comp_module, 'register_forward_pre_hook'):
                            comp_module.register_forward_pre_hook(
                                lambda m, i, comp=comp_name: self.attention_component_start_hook(m, i, comp))
                            comp_module.register_forward_hook(
                                lambda m, i, o, comp=comp_name: self.attention_component_end_hook(m, i, o, comp))
                    
                    # 新增: 添加注意力计算的钩子 - 需要找到计算注意力的具体方法或函数
                    # 这里假设SparseLlamaAttention中有一个compute_attention方法
                    if hasattr(module, 'compute_attention'):
                        original_compute_attention = module.compute_attention
                        
                        def compute_attention_wrapper(*args, **kwargs):
                            # 开始计时
                            torch.cuda.synchronize() if torch.cuda.is_available() else None
                            self._in_attention_compute = True
                            self._attention_compute_start_time = time.time()
                            
                            # 调用原始方法
                            result = original_compute_attention(*args, **kwargs)
                            
                            # 结束计时
                            torch.cuda.synchronize() if torch.cuda.is_available() else None
                            elapsed = (time.time() - self._attention_compute_start_time) * 1000  # 毫秒
                            
                            # 更新统计
                            self.attention_compute_stats["total_time"] += elapsed
                            self.attention_compute_stats["call_count"] += 1
                            
                            # 添加到组件统计中
                            self.attention_component_stats["components"]["compute_attention"]["total_time"] += elapsed
                            self.attention_component_stats["components"]["compute_attention"]["call_count"] += 1
                            
                            self._in_attention_compute = False
                            
                            return result
                        
                        # 替换原始方法
                        module.compute_attention = compute_attention_wrapper
                else:
                    module.register_forward_pre_hook(self.module_start_hook)
                    module.register_forward_hook(self.module_end_hook)
    
        # 标记钩子已注册
        self._hooks_registered = True
    
    def global_start_hook(self, module, input):
        """全局开始计时钩子"""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.global_start_time = time.time()
        
        # 记录输入信息
        if input and len(input) > 0 and hasattr(input[0], 'shape'):
            input_ids = input[0]
            self.input_shapes.append(input_ids.shape)
            
            # 计算tokens
            batch_size, seq_len = input_ids.shape[:2]
            self.global_stats["total_tokens"] += batch_size * seq_len
            self.global_stats["total_samples"] += batch_size
            
            # 判断当前阶段 (prefill 或 decode)
            # 通常prefill阶段输入序列较长，decode阶段输入为1个token
            is_prefill = seq_len > 1
            
            # 更新当前状态
            self.current_phase = "prefill" if is_prefill else "decode"
            
            # 如果是新的prefill阶段，开始一个新样本
            if is_prefill:
                self.current_sample_id = id(input_ids)
                self.current_sample_start_time = time.time()
                self.current_sample_attention_time = 0
                self.current_sample_decode_times = []
            
            # 只在需要时打印
            if self.print_per_forward:
                phase = "Prefill" if is_prefill else "Decode"
                print(f"[{phase}] 输入shape: {input_ids.shape}, tokens: {batch_size * seq_len}")
    
    def global_end_hook(self, module, input, output):
        """全局结束计时钩子"""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - self.global_start_time) * 1000  # 毫秒
        
        self.global_stats["total_time"] += elapsed
        self.global_stats["call_count"] += 1
        
        # 根据当前阶段更新统计
        if self.current_phase == "prefill":
            self.prefill_stats["total_time"] += elapsed
            self.prefill_stats["call_count"] += 1
            
            # 记录本次prefill的时间
            if self.current_sample_id is not None:
                prefill_time = elapsed
                self.prefill_stats["sample_times"].append(prefill_time)
                
                # 打印详细信息
                if self.print_per_forward:
                    attn_percentage = (self.current_sample_attention_time / prefill_time) * 100 if prefill_time > 0 else 0
                    print(f"Prefill完成: 总时间={prefill_time:.2f}ms, "
                          f"注意力层时间={self.current_sample_attention_time:.2f}ms ({attn_percentage:.1f}%)")
        
        elif self.current_phase == "decode":
            self.decode_stats["total_time"] += elapsed
            self.decode_stats["call_count"] += 1
            
            # 记录本次decode的token时间
            token_time = elapsed
            self.decode_stats["token_times"].append(token_time)
            
            # 如果有当前样本，添加到该样本的decode时间列表
            if self.current_sample_id is not None:
                self.current_sample_decode_times.append(token_time)
            
            # 检查是否是样本处理的最后一步
            is_last_token = False
            
            # 这里的判断逻辑需要根据实际模型行为调整
            if hasattr(output, 'is_last') and output.is_last:
                is_last_token = True
            elif len(self.current_sample_decode_times) >= self._max_out_len - 1:
                is_last_token = True
            
            if is_last_token and self.current_sample_id is not None:
                # 计算样本总decode时间
                total_decode_time = sum(self.current_sample_decode_times)
                self.decode_stats["samples_total_time"].append(total_decode_time)
                
                # 计算并记录样本总处理时间
                if len(self.prefill_stats["sample_times"]) > 0:
                    prefill_time = self.prefill_stats["sample_times"][-1]
                    total_time = prefill_time + total_decode_time
                    self.sample_processing_times.append(total_time)
                    
                    if self.print_per_forward:
                        print(f"样本处理完成: Prefill={prefill_time:.2f}ms, "
                              f"Decode总计={total_decode_time:.2f}ms, "
                              f"每token={total_decode_time/len(self.current_sample_decode_times):.2f}ms, "
                              f"总计={total_time:.2f}ms")
                
                # 重置当前样本状态
                self.current_sample_id = None
                self.current_sample_decode_times = []
        
        # 记录输出信息
        if hasattr(output, 'logits') and hasattr(output.logits, 'shape'):
            self.output_shapes.append(output.logits.shape)
            if self.print_per_forward:
                print(f"输出logits形状: {output.logits.shape}")
        
        if self.print_per_forward:
            print(f"本次推理时间: {elapsed:.2f}毫秒")
        
        return output
    
    def attention_start_hook(self, module, input):
        """SparseLlamaAttention层开始时的钩子"""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._in_attention_layer = True
        self._attention_start_time = time.time()
        self.module_start_times[module] = time.time()
    
    def attention_end_hook(self, module, input, output):
        """SparseLlamaAttention层结束时的钩子"""
        if not self._in_attention_layer:
            return output
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - self._attention_start_time) * 1000  # 毫秒
        
        # 记录模块统计
        module_name = f"{module.__class__.__name__}_{getattr(module, 'layer_idx', '')}"
        self.module_stats[module_name]["total_time"] += elapsed
        self.module_stats[module_name]["call_count"] += 1
        
        # 更新注意力层合并统计
        self.attention_layer_stats["total_time"] += elapsed
        self.attention_layer_stats["call_count"] += 1
        
        # 如果有layer_idx，记录到特定层的统计
        layer_idx = getattr(module, 'layer_idx', 'unknown')
        self.attention_layer_stats["layers"][layer_idx]["total_time"] += elapsed
        self.attention_layer_stats["layers"][layer_idx]["call_count"] += 1
        
        # 统计注意力层总时间
        self.attention_component_stats["total_attention_time"] += elapsed
        
        # 根据当前阶段更新注意力层时间统计
        if self.current_phase == "prefill":
            self.prefill_stats["attention_time"] += elapsed
            self.current_sample_attention_time += elapsed
        elif self.current_phase == "decode":
            self.decode_stats["attention_time"] += elapsed
        
        self._in_attention_layer = False
        
        return output
    
    def attention_component_start_hook(self, module, input, component_name):
        """SparseLlamaAttention内部组件开始计时钩子"""
        if not self._in_attention_layer:
            return
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._current_attention_component = component_name
        self._component_start_time = time.time()
    
    def attention_component_end_hook(self, module, input, output, component_name):
        """SparseLlamaAttention内部组件结束计时钩子"""
        if not self._in_attention_layer or self._current_attention_component != component_name:
            return output
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - self._component_start_time) * 1000  # 毫秒
        
        # 更新组件统计
        self.attention_component_stats["components"][component_name]["total_time"] += elapsed
        self.attention_component_stats["components"][component_name]["call_count"] += 1
        
        self._current_attention_component = None
        
        return output
    
    def module_start_hook(self, module, input):
        """模块级别开始计时钩子"""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.module_start_times[module] = time.time()
    
    def module_end_hook(self, module, input, output):
        """模块级别结束计时钩子"""
        if module not in self.module_start_times:
            return output
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - self.module_start_times[module]) * 1000  # 毫秒
        
        module_name = f"{module.__class__.__name__}"
        self.module_stats[module_name]["total_time"] += elapsed
        self.module_stats[module_name]["call_count"] += 1
        
        return output
    
    # 设置最大输出长度（用于检测样本是否处理完成）
    def set_max_out_len(self, max_out_len):
        self._max_out_len = max_out_len
    
    def report(self):
        """生成监控报告"""
        print("\n========== 模型监控报告 (batch_size=1) ==========")
        
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
            
            # 新增: 平均每条数据推理时间
            if len(self.sample_processing_times) > 0:
                avg_sample_time = np.mean(self.sample_processing_times)
                print(f"平均每条数据推理时间: {avg_sample_time:.2f}毫秒 ({avg_sample_time/1000:.2f}秒)")
        
        # 删除Prefill阶段统计
        # 删除Decode阶段统计
        
        # 整体样本处理统计
        if self.prefill_stats["sample_times"] and self.decode_stats["samples_total_time"]:
            print("\n----- 整体样本处理性能 -----")
            
            # 计算整体处理时间
            if len(self.sample_processing_times) > 0:
                total_times = np.array(self.sample_processing_times)
                avg_total_time = np.mean(total_times)
                print(f"平均每样本总处理时间: {avg_total_time:.2f}毫秒 ({avg_total_time/1000:.2f}秒)")
                
                prefill_avg = np.mean(self.prefill_stats["sample_times"])
                decode_avg = np.mean(self.decode_stats["samples_total_time"])
                
                prefill_ratio = (prefill_avg / (prefill_avg + decode_avg)) * 100
                decode_ratio = (decode_avg / (prefill_avg + decode_avg)) * 100
                
                print(f"Prefill阶段占比: {prefill_ratio:.1f}%")
                print(f"Decode阶段占比: {decode_ratio:.1f}%")
                
                # 新增: 整体每条数据token吞吐量
                avg_tokens_per_sample = len(self.decode_stats['token_times'])/len(self.decode_stats["samples_total_time"]) if len(self.decode_stats["samples_total_time"]) > 0 else 0
                end_to_end_tokens_per_second = (avg_tokens_per_sample * 1000) / avg_total_time if avg_total_time > 0 else 0
                print(f"端到端每秒token吞吐量: {end_to_end_tokens_per_second:.2f} tokens/秒")
        
        # 模块统计信息
        print("\n----- 模块性能指标 -----")
        sorted_modules = sorted(self.module_stats.items(), 
                              key=lambda x: x[1]["total_time"], 
                              reverse=True)
        
        # 分类统计模块信息
        attention_layers = []
        other_modules = []
        
        for module_name, stats in sorted_modules:
            if "SparseLlamaAttention" in module_name:
                attention_layers.append((module_name, stats))
            else:
                other_modules.append((module_name, stats))
        
        # 打印非注意力层的模块统计
        for module_name, stats in other_modules[:20]:  # 只显示前20个耗时最长的非注意力模块
            avg_time = stats["total_time"] / stats["call_count"] if stats["call_count"] > 0 else 0
            time_percent = (stats["total_time"] / total_time * 100) if total_time > 0 else 0
            
            print(f"{module_name}: 总计 {stats['total_time']:.2f}ms, "
                  f"调用 {stats['call_count']}次, "
                  f"平均 {avg_time:.2f}ms/次, "
                  f"占比 {time_percent:.1f}%")
        
        # 打印注意力层的合并统计
        if attention_layers:
            print("\n----- SparseLlamaAttention 总体性能 -----")
            total_attention_time = self.attention_layer_stats["total_time"]
            total_attention_calls = self.attention_layer_stats["call_count"]
            attention_avg_time = total_attention_time / total_attention_calls if total_attention_calls > 0 else 0
            attention_percent = (total_attention_time / total_time * 100) if total_time > 0 else 0
            
            print(f"SparseLlamaAttention 所有层: 总计 {total_attention_time:.2f}ms, "
                  f"调用 {total_attention_calls}次, "
                  f"平均 {attention_avg_time:.2f}ms/次, "
                  f"占比 {attention_percent:.1f}%")
        
        # 新增: SparseLlamaAttention内部组件统计
        if self.attention_component_stats["total_attention_time"] > 0:
            print("\n----- SparseLlamaAttention内部组件性能 -----")
            total_attn_time = self.attention_component_stats["total_attention_time"]
            print(f"SparseLlamaAttention总时间: {total_attn_time:.2f}ms")
            print(f"占全局推理时间比例: {(total_attn_time / total_time * 100):.1f}%")
            
            sorted_components = sorted(
                self.attention_component_stats["components"].items(),
                key=lambda x: x[1]["total_time"],
                reverse=True
            )
            
            print("\n组件占用时间百分比:")
            for comp_name, stats in sorted_components:
                comp_time = stats["total_time"]
                comp_calls = stats["call_count"]
                comp_percent_of_attn = (comp_time / total_attn_time * 100) if total_attn_time > 0 else 0
                comp_percent_of_total = (comp_time / total_time * 100) if total_time > 0 else 0
                
                print(f"{comp_name}: 总计 {comp_time:.2f}ms, "
                      f"调用 {comp_calls}次, "
                      f"占注意力层时间 {comp_percent_of_attn:.1f}%, "
                      f"占全局时间 {comp_percent_of_total:.1f}%")
                
            # 新增: 添加注意力计算占用时间百分比
            if self.attention_compute_stats["call_count"] > 0:
                compute_time = self.attention_compute_stats["total_time"]
                compute_calls = self.attention_compute_stats["call_count"]
                compute_percent_of_attn = (compute_time / total_attn_time * 100) if total_attn_time > 0 else 0
                compute_percent_of_total = (compute_time / total_time * 100) if total_time > 0 else 0
                
                print(f"compute_attention: 总计 {compute_time:.2f}ms, "
                      f"调用 {compute_calls}次, "
                      f"占注意力层时间 {compute_percent_of_attn:.1f}%, "
                      f"占全局时间 {compute_percent_of_total:.1f}%")
        
        # 输入输出统计
        print("\n----- 输入/输出统计 -----")
        print(f"输入形状示例: {self.input_shapes[-5:] if self.input_shapes else 'None'}")
        print(f"输出形状示例: {self.output_shapes[-5:] if self.output_shapes else 'None'}")
        
        print("\n===============================")
    
    def reset(self):
        """重置所有统计信息"""
        # 重置全局统计
        self.global_stats = {
            "total_time": 0,
            "call_count": 0,
            "total_tokens": 0,
            "total_samples": 0
        }
        
        # 重置模块统计
        self.module_start_times = {}
        self.module_stats = defaultdict(lambda: {"total_time": 0, "call_count": 0})
        
        # 重置输入输出监控
        self.input_shapes = []
        self.output_shapes = []
        
        # 重置样本处理时间
        self.sample_processing_times = []
        
        # 重置prefill和decode统计
        self.prefill_stats = {
            "total_time": 0,
            "call_count": 0,
            "attention_time": 0,
            "sample_times": []
        }
        
        self.decode_stats = {
            "total_time": 0,
            "call_count": 0,
            "attention_time": 0,
            "token_times": [],
            "samples_total_time": []
        }
        
        # 重置当前状态
        self.current_sample_id = None
        self.current_phase = None
        self.current_sample_start_time = None
        self.current_sample_attention_time = 0
        self.current_sample_decode_times = []
        
        # 重置内部状态
        self._in_attention_layer = False
        self._attention_start_time = None
        
        # 重置注意力层组件统计
        self.attention_component_stats = {
            "total_attention_time": 0,
            "components": defaultdict(lambda: {"total_time": 0, "call_count": 0})
        }
        self._current_attention_component = None
        self._component_start_time = None
        
        # 重置注意力层合并统计
        self.attention_layer_stats = {
            "total_time": 0,
            "call_count": 0,
            "layers": defaultdict(lambda: {"total_time": 0, "call_count": 0})
        }
        
        # 重置注意力计算统计
        self.attention_compute_stats = {
            "total_time": 0,
            "call_count": 0
        }
        self._in_attention_compute = False
        self._attention_compute_start_time = None
        
        print("已重置所有监控统计信息")

# 全局监控器实例
global_monitor = ModelMonitor(print_per_forward=False)