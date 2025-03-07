import sys
import time
from collections import defaultdict

import numpy as np
import torch


class ModelMonitor:

    def __init__(self, print_per_forward=False):
        # 基本设置
        self.print_per_forward = print_per_forward

        # 全局级别监控
        self.global_stats = {
            'total_time': 0,
            'call_count': 0,
            'total_tokens': 0,
            'total_samples': 0
        }

        # 模块级别监控
        self.module_start_times = {}
        self.module_stats = defaultdict(lambda: {
            'total_time': 0,
            'call_count': 0
        })

        # 输入输出监控
        self.input_shapes = []
        self.output_shapes = []

        # 样本处理时间
        self.sample_processing_times = []

        # 输入token统计
        self.input_token_stats = {
            'samples': [],  # 每个样本的输入token数
            'prefill_tokens': [],  # prefill阶段的token数
            'decode_tokens': [],  # decode阶段的token数
            'total_tokens': 0,  # 所有token总数
        }

        # prefill 和 decode 阶段监控
        self.prefill_stats = {
            'total_time': 0,
            'call_count': 0,
            'attention_time': 0,  # 注意力层耗时
            'sample_times': []
        }

        self.decode_stats = {
            'total_time': 0,
            'call_count': 0,
            'attention_time': 0,  # 注意力层耗时
            'token_times': [],
            'samples_total_time': []
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

        # 注意力层内部组件统计
        self.attention_component_stats = {
            'total_attention_time': 0,
            'components': defaultdict(lambda: {
                'total_time': 0,
                'call_count': 0
            })
        }
        self._current_attention_component = None
        self._component_start_time = None

        # 注意力层合并统计
        self.attention_layer_stats = {
            'total_time': 0,
            'call_count': 0,
            'layers': defaultdict(lambda: {
                'total_time': 0,
                'call_count': 0
            })
        }

        # 注意力计算统计
        self.attention_compute_stats = {'total_time': 0, 'call_count': 0}
        self._in_attention_compute = False
        self._attention_compute_start_time = None

    def register_hooks(self, model):
        """为模型和其子模块注册监控钩子."""
        # 检查模型是否支持钩子
        if hasattr(model, 'register_forward_pre_hook'):
            # 为整个模型注册全局钩子
            model.register_forward_pre_hook(self.global_start_hook)
            model.register_forward_hook(self.global_end_hook)

        # 为所有子模块注册计时钩子
        for name, module in model.named_modules():
            if name and hasattr(module,
                                'register_forward_pre_hook'):  # 检查模块是否支持钩子
                # 为SparseLlamaAttention特殊处理
                if module.__class__.__name__ == 'SparseLlamaAttention':
                    module.register_forward_pre_hook(self.attention_start_hook)
                    module.register_forward_hook(self.attention_end_hook)

                    # 为SparseLlamaAttention内部组件注册钩子
                    for comp_name, comp_module in module.named_children():
                        if hasattr(comp_module, 'register_forward_pre_hook'):
                            comp_module.register_forward_pre_hook(
                                lambda m, i, comp=comp_name: self.
                                attention_component_start_hook(m, i, comp))
                            comp_module.register_forward_hook(
                                lambda m, i, o, comp=comp_name: self.
                                attention_component_end_hook(m, i, o, comp))

                    # 添加注意力计算的钩子
                    if hasattr(module, 'compute_attention'):
                        original_compute_attention = module.compute_attention

                        def compute_attention_wrapper(*args, **kwargs):
                            # 开始计时
                            torch.cuda.synchronize(
                            ) if torch.cuda.is_available() else None
                            self._in_attention_compute = True
                            self._attention_compute_start_time = time.time()

                            # 调用原始方法
                            result = original_compute_attention(
                                *args, **kwargs)

                            # 结束计时
                            torch.cuda.synchronize(
                            ) if torch.cuda.is_available() else None
                            elapsed = (time.time() -
                                       self._attention_compute_start_time
                                       ) * 1000  # 毫秒

                            # 更新统计
                            self.attention_compute_stats[
                                'total_time'] += elapsed
                            self.attention_compute_stats['call_count'] += 1

                            # 添加到组件统计中
                            self.attention_component_stats['components'][
                                'compute_attention']['total_time'] += elapsed
                            self.attention_component_stats['components'][
                                'compute_attention']['call_count'] += 1

                            self._in_attention_compute = False

                            return result

                        # 替换原始方法
                        module.compute_attention = compute_attention_wrapper
                else:
                    module.register_forward_pre_hook(self.module_start_hook)
                    module.register_forward_hook(self.module_end_hook)

    def global_start_hook(self, module, input):
        """全局开始计时钩子."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.global_start_time = time.time()

        # 记录输入信息
        if input and len(input) > 0 and hasattr(input[0], 'shape'):
            input_ids = input[0]
            self.input_shapes.append(input_ids.shape)

            # 计算tokens
            batch_size, seq_len = input_ids.shape[:2]
            current_tokens = batch_size * seq_len
            self.global_stats['total_tokens'] += current_tokens
            self.global_stats['total_samples'] += batch_size

            # 统计输入token
            self.input_token_stats['total_tokens'] += current_tokens

            # 判断当前阶段 (prefill 或 decode)
            is_prefill = seq_len > 1

            if is_prefill:
                self.input_token_stats['prefill_tokens'].append(current_tokens)
                # 新样本开始
                self.current_sample_id = id(input_ids)
                self.current_sample_start_time = time.time()
                self.current_sample_attention_time = 0
                self.current_sample_decode_times = []
                # 初始化样本token计数
                self.input_token_stats['samples'].append({
                    'prefill': current_tokens,
                    'decode': 0
                })
            else:
                self.input_token_stats['decode_tokens'].append(current_tokens)
                # 更新当前样本的decode token计数
                if self.input_token_stats['samples']:
                    self.input_token_stats['samples'][-1][
                        'decode'] += current_tokens

            # 更新当前状态
            self.current_phase = 'prefill' if is_prefill else 'decode'

            # 只在需要时打印
            if self.print_per_forward:
                phase = 'Prefill' if is_prefill else 'Decode'
                print(
                    f'[{phase}] 输入shape: {input_ids.shape}, tokens: {current_tokens}'
                )

    def global_end_hook(self, module, input, output):
        """全局结束计时钩子."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - self.global_start_time) * 1000  # 毫秒

        self.global_stats['total_time'] += elapsed
        self.global_stats['call_count'] += 1

        # 根据当前阶段更新统计
        if self.current_phase == 'prefill':
            self.prefill_stats['total_time'] += elapsed
            self.prefill_stats['call_count'] += 1

            # 记录本次prefill的时间
            if self.current_sample_id is not None:
                prefill_time = elapsed
                self.prefill_stats['sample_times'].append(prefill_time)

                # 打印详细信息
                if self.print_per_forward:
                    attn_percentage = (
                        self.current_sample_attention_time /
                        prefill_time) * 100 if prefill_time > 0 else 0
                    print(
                        f'Prefill完成: 总时间={prefill_time:.2f}ms, '
                        f'注意力层时间={self.current_sample_attention_time:.2f}ms ({attn_percentage:.1f}%)'
                    )

        elif self.current_phase == 'decode':
            self.decode_stats['total_time'] += elapsed
            self.decode_stats['call_count'] += 1

            # 记录本次decode的token时间
            token_time = elapsed
            self.decode_stats['token_times'].append(token_time)

            # 如果有当前样本，添加到该样本的decode时间列表
            if self.current_sample_id is not None:
                self.current_sample_decode_times.append(token_time)

            # 检查是否是样本处理的最后一步
            is_last_token = False

            # 这里的判断逻辑需要根据实际模型行为调整
            if hasattr(output, 'is_last') and output.is_last:
                is_last_token = True
            elif len(
                    self.current_sample_decode_times) >= self._max_out_len - 1:
                is_last_token = True

            if is_last_token and self.current_sample_id is not None:
                # 计算样本总decode时间
                total_decode_time = sum(self.current_sample_decode_times)
                self.decode_stats['samples_total_time'].append(
                    total_decode_time)

                # 计算并记录样本总处理时间
                if len(self.prefill_stats['sample_times']) > 0:
                    prefill_time = self.prefill_stats['sample_times'][-1]
                    total_time = prefill_time + total_decode_time
                    self.sample_processing_times.append(total_time)

                    if self.print_per_forward:
                        print(
                            f'样本处理完成: Prefill={prefill_time:.2f}ms, '
                            f'Decode总计={total_decode_time:.2f}ms, '
                            f'每token={total_decode_time/len(self.current_sample_decode_times):.2f}ms, '
                            f'总计={total_time:.2f}ms')

                # 重置当前样本状态
                self.current_sample_id = None
                self.current_sample_decode_times = []

        # 记录输出信息
        if hasattr(output, 'logits') and hasattr(output.logits, 'shape'):
            self.output_shapes.append(output.logits.shape)
            if self.print_per_forward:
                print(f'输出logits形状: {output.logits.shape}')

        if self.print_per_forward:
            print(f'本次推理时间: {elapsed:.2f}毫秒')

        return output

    def attention_start_hook(self, module, input):
        """SparseLlamaAttention层开始时的钩子."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._in_attention_layer = True
        self._attention_start_time = time.time()
        self.module_start_times[module] = time.time()

    def attention_end_hook(self, module, input, output):
        """SparseLlamaAttention层结束时的钩子."""
        if not self._in_attention_layer:
            return output

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - self._attention_start_time) * 1000  # 毫秒

        # 记录模块统计
        module_name = f"{module.__class__.__name__}_{getattr(module, 'layer_idx', '')}"
        self.module_stats[module_name]['total_time'] += elapsed
        self.module_stats[module_name]['call_count'] += 1

        # 更新注意力层合并统计
        self.attention_layer_stats['total_time'] += elapsed
        self.attention_layer_stats['call_count'] += 1

        # 如果有layer_idx，记录到特定层的统计
        layer_idx = getattr(module, 'layer_idx', 'unknown')
        self.attention_layer_stats['layers'][layer_idx][
            'total_time'] += elapsed
        self.attention_layer_stats['layers'][layer_idx]['call_count'] += 1

        # 统计注意力层总时间
        self.attention_component_stats['total_attention_time'] += elapsed

        # 根据当前阶段更新注意力层时间统计
        if self.current_phase == 'prefill':
            self.prefill_stats['attention_time'] += elapsed
            self.current_sample_attention_time += elapsed
        elif self.current_phase == 'decode':
            self.decode_stats['attention_time'] += elapsed

        self._in_attention_layer = False

        return output

    def attention_component_start_hook(self, module, input, component_name):
        """SparseLlamaAttention内部组件开始计时钩子."""
        if not self._in_attention_layer:
            return

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._current_attention_component = component_name
        self._component_start_time = time.time()

    def attention_component_end_hook(self, module, input, output,
                                     component_name):
        """SparseLlamaAttention内部组件结束计时钩子."""
        if not self._in_attention_layer or self._current_attention_component != component_name:
            return output

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - self._component_start_time) * 1000  # 毫秒

        # 更新组件统计
        self.attention_component_stats['components'][component_name][
            'total_time'] += elapsed
        self.attention_component_stats['components'][component_name][
            'call_count'] += 1

        self._current_attention_component = None

        return output

    def module_start_hook(self, module, input):
        """模块级别开始计时钩子."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.module_start_times[module] = time.time()

    def module_end_hook(self, module, input, output):
        """模块级别结束计时钩子."""
        if module not in self.module_start_times:
            return output

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - self.module_start_times[module]) * 1000  # 毫秒

        module_name = f'{module.__class__.__name__}'
        self.module_stats[module_name]['total_time'] += elapsed
        self.module_stats[module_name]['call_count'] += 1

        return output

    def set_max_out_len(self, max_out_len):
        """设置最大输出长度."""
        self._max_out_len = max_out_len

    def report(self):
        """生成监控报告."""
        print('\n========== 模型监控报告 (batch_size=1) ==========')

        # 全局统计信息
        print('\n----- 全局性能指标 -----')
        total_time = self.global_stats['total_time']
        call_count = self.global_stats['call_count']
        if call_count > 0:
            avg_time = total_time / call_count
            tokens_per_sec = self.global_stats['total_tokens'] / (
                total_time / 1000) if total_time > 0 else 0

            print(f'总推理次数: {call_count}')
            print(f'总推理时间: {total_time:.2f}毫秒 ({total_time/1000:.2f}秒)')
            print(f'平均推理时间: {avg_time:.2f}毫秒/次')
            print(f"处理样本总数: {self.global_stats['total_samples']}")
            print(f"处理token总数: {self.global_stats['total_tokens']}")
            print(f'吞吐量: {tokens_per_sec:.2f} tokens/秒')

            if len(self.sample_processing_times) > 0:
                avg_sample_time = np.mean(self.sample_processing_times)
                print(
                    f'平均每条数据推理时间: {avg_sample_time:.2f}毫秒 ({avg_sample_time/1000:.2f}秒)'
                )

        # 输入token统计
        print('\n----- 输入Token统计 -----')
        print(f"总Token数: {self.input_token_stats['total_tokens']}")

        if self.input_token_stats['prefill_tokens']:
            avg_prefill = sum(self.input_token_stats['prefill_tokens']) / len(
                self.input_token_stats['prefill_tokens'])
            print(f'Prefill阶段平均Token数: {avg_prefill:.2f}')
            print(
                f"Prefill阶段Token数分布: {self.input_token_stats['prefill_tokens']}"
            )

        if self.input_token_stats['decode_tokens']:
            avg_decode = sum(self.input_token_stats['decode_tokens']) / len(
                self.input_token_stats['decode_tokens'])
            print(f'Decode阶段平均Token数: {avg_decode:.2f}')

        print('\n每个样本的Token统计:')
        for i, sample in enumerate(self.input_token_stats['samples']):
            print(f"样本 {i+1}: Prefill tokens={sample['prefill']}, "
                  f"Decode tokens={sample['decode']}, "
                  f"总计={sample['prefill'] + sample['decode']}")

        # 整体样本处理统计
        if self.prefill_stats['sample_times'] and self.decode_stats[
                'samples_total_time']:
            print('\n----- 整体样本处理性能 -----')

            if len(self.sample_processing_times) > 0:
                total_times = np.array(self.sample_processing_times)
                avg_total_time = np.mean(total_times)
                print(
                    f'平均每样本总处理时间: {avg_total_time:.2f}毫秒 ({avg_total_time/1000:.2f}秒)'
                )

                prefill_avg = np.mean(self.prefill_stats['sample_times'])
                decode_avg = np.mean(self.decode_stats['samples_total_time'])

                prefill_ratio = (prefill_avg /
                                 (prefill_avg + decode_avg)) * 100
                decode_ratio = (decode_avg / (prefill_avg + decode_avg)) * 100

                print(f'Prefill阶段占比: {prefill_ratio:.1f}%')
                print(f'Decode阶段占比: {decode_ratio:.1f}%')

                avg_tokens_per_sample = len(
                    self.decode_stats['token_times']) / len(
                        self.decode_stats['samples_total_time']) if len(
                            self.decode_stats['samples_total_time']) > 0 else 0
                end_to_end_tokens_per_second = (
                    avg_tokens_per_sample *
                    1000) / avg_total_time if avg_total_time > 0 else 0
                print(
                    f'端到端每秒token吞吐量: {end_to_end_tokens_per_second:.2f} tokens/秒'
                )

        # 模块统计信息
        print('\n----- 模块性能指标 -----')
        sorted_modules = sorted(self.module_stats.items(),
                                key=lambda x: x[1]['total_time'],
                                reverse=True)

        attention_layers = []
        other_modules = []

        for module_name, stats in sorted_modules:
            if 'SparseLlamaAttention' in module_name:
                attention_layers.append((module_name, stats))
            else:
                other_modules.append((module_name, stats))

        for module_name, stats in other_modules[:20]:
            avg_time = stats['total_time'] / stats['call_count'] if stats[
                'call_count'] > 0 else 0
            time_percent = (stats['total_time'] / total_time *
                            100) if total_time > 0 else 0

            print(f"{module_name}: 总计 {stats['total_time']:.2f}ms, "
                  f"调用 {stats['call_count']}次, "
                  f'平均 {avg_time:.2f}ms/次, '
                  f'占比 {time_percent:.1f}%')

        if attention_layers:
            print('\n----- SparseLlamaAttention 总体性能 -----')
            total_attention_time = self.attention_layer_stats['total_time']
            total_attention_calls = self.attention_layer_stats['call_count']
            attention_avg_time = total_attention_time / total_attention_calls if total_attention_calls > 0 else 0
            attention_percent = (total_attention_time / total_time *
                                 100) if total_time > 0 else 0

            print(
                f'SparseLlamaAttention 所有层: 总计 {total_attention_time:.2f}ms, '
                f'调用 {total_attention_calls}次, '
                f'平均 {attention_avg_time:.2f}ms/次, '
                f'占比 {attention_percent:.1f}%')

        if self.attention_component_stats['total_attention_time'] > 0:
            print('\n----- SparseLlamaAttention内部组件性能 -----')
            total_attn_time = self.attention_component_stats[
                'total_attention_time']
            print(f'SparseLlamaAttention总时间: {total_attn_time:.2f}ms')
            print(f'占全局推理时间比例: {(total_attn_time / total_time * 100):.1f}%')

            sorted_components = sorted(
                self.attention_component_stats['components'].items(),
                key=lambda x: x[1]['total_time'],
                reverse=True)

            print('\n组件占用时间百分比:')
            for comp_name, stats in sorted_components:
                comp_time = stats['total_time']
                comp_calls = stats['call_count']
                comp_percent_of_attn = (comp_time / total_attn_time *
                                        100) if total_attn_time > 0 else 0
                comp_percent_of_total = (comp_time / total_time *
                                         100) if total_time > 0 else 0

                print(f'{comp_name}: 总计 {comp_time:.2f}ms, '
                      f'调用 {comp_calls}次, '
                      f'占注意力层时间 {comp_percent_of_attn:.1f}%, '
                      f'占全局时间 {comp_percent_of_total:.1f}%')

        # 输入输出统计
        print('\n----- 输入/输出统计 -----')
        print(
            f"输入形状示例: {self.input_shapes[-5:] if self.input_shapes else 'None'}"
        )
        print(
            f"输出形状示例: {self.output_shapes[-5:] if self.output_shapes else 'None'}"
        )

        print('\n===============================')

    def reset(self):
        """重置所有统计信息."""
        # 重置全局统计
        self.global_stats = {
            'total_time': 0,
            'call_count': 0,
            'total_tokens': 0,
            'total_samples': 0
        }

        # 重置模块统计
        self.module_start_times = {}
        self.module_stats = defaultdict(lambda: {
            'total_time': 0,
            'call_count': 0
        })

        # 重置输入输出监控
        self.input_shapes = []
        self.output_shapes = []

        # 重置样本处理时间
        self.sample_processing_times = []

        # 重置token统计
        self.input_token_stats = {
            'samples': [],
            'prefill_tokens': [],
            'decode_tokens': [],
            'total_tokens': 0,
        }

        # 重置prefill和decode统计
        self.prefill_stats = {
            'total_time': 0,
            'call_count': 0,
            'attention_time': 0,
            'sample_times': []
        }

        self.decode_stats = {
            'total_time': 0,
            'call_count': 0,
            'attention_time': 0,
            'token_times': [],
            'samples_total_time': []
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
            'total_attention_time': 0,
            'components': defaultdict(lambda: {
                'total_time': 0,
                'call_count': 0
            })
        }
        self._current_attention_component = None
        self._component_start_time = None

        # 重置注意力层合并统计
        self.attention_layer_stats = {
            'total_time': 0,
            'call_count': 0,
            'layers': defaultdict(lambda: {
                'total_time': 0,
                'call_count': 0
            })
        }

        # 重置注意力计算统计
        self.attention_compute_stats = {'total_time': 0, 'call_count': 0}
        self._in_attention_compute = False
        self._attention_compute_start_time = None

        print('已重置所有监控统计信息')

    def save_report(self, filepath):
        import json

        # 构建JSON数据结构
        report_data = {
            'basic_metrics': {
                'total_inference_count':
                self.global_stats['call_count'],
                'total_inference_time_ms':
                round(self.global_stats['total_time'], 2),
                'total_inference_time_s':
                round(self.global_stats['total_time'] / 1000, 2),
                'total_samples':
                self.global_stats['total_samples'],
                'total_tokens':
                self.global_stats['total_tokens']
            },
            'sample_statistics': [{
                'sample_id':
                i + 1,
                'prefill_tokens':
                sample['prefill'],
                'decode_tokens':
                sample['decode'],
                'total_tokens':
                sample['prefill'] + sample['decode']
            } for i, sample in enumerate(self.input_token_stats['samples'])],
            'performance_statistics': {},
            'module_statistics': {
                module_name: {
                    'total_time_ms':
                    round(stats['total_time'], 2),
                    'call_count':
                    stats['call_count'],
                    'avg_time_ms':
                    round(stats['total_time'] / stats['call_count'], 2)
                    if stats['call_count'] > 0 else 0,
                    'time_percentage':
                    round(
                        (stats['total_time'] /
                         self.global_stats['total_time'] *
                         100), 1) if self.global_stats['total_time'] > 0 else 0
                }
                for module_name, stats in sorted(
                    self.module_stats.items(),
                    key=lambda x: x[1]['total_time'],
                    reverse=True)
            },
            'attention_statistics': {
                'total_time_ms':
                round(self.attention_component_stats['total_attention_time'],
                      2),
                'time_percentage':
                round((self.attention_component_stats['total_attention_time'] /
                       self.global_stats['total_time'] *
                       100), 1) if self.global_stats['total_time'] > 0 else 0,
                'components': {
                    comp_name: {
                        'total_time_ms':
                        round(stats['total_time'], 2),
                        'call_count':
                        stats['call_count'],
                        'percentage_of_attention':
                        round((
                            stats['total_time'] / self.
                            attention_component_stats['total_attention_time'] *
                            100), 1) if
                        self.attention_component_stats['total_attention_time']
                        > 0 else 0,
                        'percentage_of_total':
                        round((stats['total_time'] /
                               self.global_stats['total_time'] * 100), 1)
                        if self.global_stats['total_time'] > 0 else 0
                    }
                    for comp_name, stats in
                    self.attention_component_stats['components'].items()
                }
            },
            'shape_statistics': {
                'recent_input_shapes':
                self.input_shapes[-5:] if self.input_shapes else None,
                'recent_output_shapes':
                self.output_shapes[-5:] if self.output_shapes else None,
                'all_input_shapes':
                self.input_shapes,
                'all_output_shapes':
                self.output_shapes
            },
            'raw_data': {
                'prefill_tokens': self.input_token_stats['prefill_tokens'],
                'decode_tokens': self.input_token_stats['decode_tokens'],
                'prefill_times_ms': self.prefill_stats['sample_times'],
                'decode_token_times_ms': self.decode_stats['token_times'],
                'sample_total_times_ms': self.sample_processing_times
            }
        }

        # 如果有处理性能统计数据，添加到report_data中
        if self.prefill_stats['sample_times'] and self.decode_stats[
                'samples_total_time'] and len(
                    self.sample_processing_times) > 0:
            total_times = np.array(self.sample_processing_times)
            avg_total_time = np.mean(total_times)
            prefill_avg = np.mean(self.prefill_stats['sample_times'])
            decode_avg = np.mean(self.decode_stats['samples_total_time'])

            report_data['performance_statistics'] = {
                'avg_sample_process_time_ms':
                round(avg_total_time, 2),
                'avg_sample_process_time_s':
                round(avg_total_time / 1000, 2),
                'prefill_phase_ratio':
                round((prefill_avg / (prefill_avg + decode_avg)) * 100, 1),
                'decode_phase_ratio':
                round((decode_avg / (prefill_avg + decode_avg)) * 100, 1)
            }

        # 保存为JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        print(f'监控报告已保存到: {filepath}')


# 全局监控器实例
global_monitor = ModelMonitor(print_per_forward=False)

import json
import os
from datetime import datetime


class TokenCounter:

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.history = []  # 用于记录每次的token使用情况

    def update(self, input_count, output_count):
        self.total_input_tokens += input_count
        self.total_output_tokens += output_count

        # 记录本次使用情况
        self.history.append({
            'timestamp':
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_tokens':
            input_count,
            'output_tokens':
            output_count
        })

    def get_totals(self):
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens
        }

    def save_to_file(self, filepath='token_usage_stats.json'):
        # 准备要保存的数据
        data = {
            'summary': self.get_totals(),
            'history': self.history,
            'save_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 确保目录存在
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f'Token usage statistics saved to {filepath}')


# 创建全局计数器实例
token_counter = TokenCounter()
