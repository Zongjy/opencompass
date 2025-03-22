import math
import warnings
from tkinter import NO
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (apply_rotary_pos_emb,
                                                      logger, repeat_kv)
from transformers.utils import logging

from .apply_rope import triton_apply_rotary_pos_emb
from .cake.cake_cache import CakeCache, CakeDecodingKVCache_LayerWise
from .cake.utils import calculate_entropy
from .flex_prefill_attention import flex_prefill_attention
from .pyramidkv_utils import (DynamicCacheSplitHeadFlatten, init_adakv,
                              init_CAM, init_H2O, init_headkv, init_l2norm,
                              init_pyramidkv, init_snapkv, init_sparq,
                              init_StreamingLLM)

logger = logging.get_logger(__name__)


def _flash_attention_forward(self,
                             query_states,
                             key_states,
                             value_states,
                             attention_mask,
                             query_length,
                             dropout=0.0,
                             softmax_scale=None):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
    """
    # if not self._flash_attn_uses_top_left_mask:
    #     causal = self.is_causal
    # else:
    # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
    causal = self.is_causal and query_length != 1

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            query_states, key_states, value_states, attention_mask,
            query_length)

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )

        attn_output = pad_input(attn_output_unpad, indices_q, batch_size,
                                query_length)
    else:
        attn_output = flash_attn_func(query_states,
                                      key_states,
                                      value_states,
                                      dropout,
                                      softmax_scale=softmax_scale,
                                      causal=causal)

    # if self.layer_idx == 0:
    #     import pdb; pdb.set_trace()

    return attn_output


import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def llama_keyformer_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    # KeyFormer相关参数
    keyformer: bool = True,
    tau_init: float = 1.0,
    tau_delta: float = 0.01,
    kv_cache: float = 60.0,
    recent: float = 30.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    """支持KeyFormer的Llama注意力前向传播实现.

    额外参数:
    keyformer: 是否启用KeyFormer功能
    tau_init: Gumbel-softmax初始温度参数
    tau_delta: Gumbel-softmax温度增量
    kv_cache: 要保留的KV缓存百分比
    recent: 最近tokens在保留缓存中的百分比
    """
    if output_attentions:
        logger.warning_once(
            'KeyFormer does not support `output_attentions=True`. Falling back to standard attention.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    init_H2O(self)

    bsz, q_len, _ = hidden_states.size()

    # 初始化/更新KeyFormer状态计数器
    if not hasattr(self, 'keyformer_itr_count'):
        self.keyformer_itr_count = 0
        self.keyformer_score_fn = None
        self.keyformer_token_discard_mask = None
        self.keyformer_token_discard_idx = None
        self.keyformer_req_tokens = None

    # 投影查询、键和值
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # 重塑张量以进行多头注意力计算
    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-1]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(f'需要初始化层索引以使用KV缓存')
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

    # 应用旋转位置编码
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)

    # 处理多查询注意力
    key_states_original = key_states
    value_states_original = value_states
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # 处理过去的KV缓存
    if past_key_value is not None and len(past_key_value) != 0:
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }

        # 标准的KV缓存更新
        key_prev, value_prev = past_key_value.get_past_keys_values(
            self.layer_idx)

        if key_prev is not None and value_prev is not None:
            key_states = torch.cat([key_prev, key_states], dim=2)
            value_states = torch.cat([value_prev, value_states], dim=2)

        # 更新或设置kv_seq_len
        self.kv_seq_len = key_states.shape[2]

    # 创建因果掩码
    min_val = torch.finfo(query_states.dtype).min
    causal_mask = None

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key_states.shape[2]]
    elif q_len > 1:  # 自动创建因果掩码
        s = max(q_len, key_states.shape[2])
        causal_mask = query_states.new_ones(1, 1, s, s, dtype=torch.float16)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[:, :, -q_len:, -key_states.shape[2]:]

    # 计算注意力分数
    attn_scale = 1.0 / math.sqrt(self.head_dim)
    attn_weights = torch.matmul(query_states, key_states.transpose(
        -2, -1)) * attn_scale

    # 应用掩码
    if causal_mask is not None:
        attn_weights = attn_weights.masked_fill(causal_mask, min_val)

    # KeyFormer: Gumbel-Softmax部分
    gumbel_score = None
    if keyformer and q_len > 1:  # 在提示阶段使用标准注意力
        # 在自回归生成阶段应用KeyFormer
        self.keyformer_itr_count = 0  # 在提示阶段重置
        self.keyformer_score_fn = None
        self.keyformer_req_tokens = None
    elif keyformer:
        # 计算当前的tau值
        current_tau = tau_init + (self.keyformer_itr_count * tau_delta)

        # 应用Gumbel-Softmax
        gumbel_score = F.gumbel_softmax(attn_weights,
                                        tau=current_tau,
                                        hard=False,
                                        dim=-1)

    # 应用Softmax得到标准注意力权重
    attn_weights = F.softmax(attn_weights.float(),
                             dim=-1).to(query_states.dtype)

    # 对注意力权重应用dropout
    if self.training and self.attention_dropout > 0:
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

    # 聚合KeyFormer分数并管理KV缓存
    if keyformer and q_len == 1:  # 在自回归生成阶段
        # 合并Gumbel注意力分数
        current_score_fn = gumbel_score.sum(0).sum(1)  # (heads, kv_len)

        # 累积注意力分数
        if self.keyformer_score_fn is not None:
            # 将当前分数添加到过去的分数中（保持最后一个token的分数不变）
            current_score_fn[:, :-1] += self.keyformer_score_fn
        else:
            # 初始化所需的tokens数量
            total_tokens = current_score_fn.shape[-1]
            req_tokens = int((total_tokens * kv_cache) / 100)
            recent_tokens = int((req_tokens * recent) / 100)
            key_tokens = req_tokens - recent_tokens
            self.keyformer_req_tokens = (recent_tokens, key_tokens)

        # 更新累积分数
        self.keyformer_score_fn = current_score_fn

        # 每8个迭代周期执行KV缓存压缩
        if self.keyformer_itr_count % 8 == 0 and self.keyformer_score_fn is not None:
            # 创建token掩码
            token_mask = torch.ones(self.keyformer_score_fn.shape[0],
                                    self.keyformer_score_fn.shape[1] + 1,
                                    device=attn_weights.device,
                                    dtype=torch.bool)

            # 获取要保留的tokens的总数
            recent_tokens, key_tokens = self.keyformer_req_tokens
            total_tokens = self.keyformer_score_fn.shape[-1]

            # 如果当前缓存大于所需的缓存
            if total_tokens > (recent_tokens + key_tokens):
                # 保留最近的tokens
                if recent_tokens > 0:
                    token_mask[:, :-recent_tokens] = 0
                    key_tokens_window = self.keyformer_score_fn[:, :
                                                                -recent_tokens]
                else:
                    key_tokens_window = self.keyformer_score_fn

                # 保留重要的key tokens
                if key_tokens > 0:
                    _, keep_topk = key_tokens_window.topk(k=key_tokens,
                                                          dim=-1,
                                                          largest=True)
                    token_mask = token_mask.scatter(-1, keep_topk, 1)

            # 计算稀疏长度
            sparse_len = recent_tokens + key_tokens

            # 对KV缓存应用掩码
            # 首先，处理键
            k_shape = key_states.shape
            key_states_flat = key_states.transpose(-2, -1).reshape(
                bsz * self.num_heads, -1, self.head_dim)
            key_states_masked = key_states_flat[token_mask.repeat(bsz, 1, 1)]
            key_states_new = key_states_masked.reshape(
                bsz, self.num_heads, sparse_len,
                self.head_dim).transpose(-2, -1)

            # 然后，处理值
            value_states_flat = value_states.reshape(bsz * self.num_heads, -1,
                                                     self.head_dim)
            value_states_masked = value_states_flat[token_mask.repeat(
                bsz, 1, 1)]
            value_states_new = value_states_masked.reshape(
                bsz, self.num_heads, sparse_len, self.head_dim)

            # 更新键和值
            key_states = key_states_new
            value_states = value_states_new

            # 更新累积分数，只保留未丢弃的tokens
            self.keyformer_score_fn = self.keyformer_score_fn[
                token_mask[:, :-1]]
            self.keyformer_score_fn = self.keyformer_score_fn.reshape(
                self.num_key_value_heads, -1)

    # 计算注意力输出
    attn_output = torch.matmul(attn_weights, value_states)

    # 整理输出形状
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(
        bsz, q_len, -1)

    # 应用输出投影
    attn_output = self.o_proj(attn_output)

    # 更新KV缓存（如果需要）
    if past_key_value is not None and use_cache:
        # 标准KV缓存更新
        past_key_value.update(key_states, value_states, self.layer_idx, {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        })

        # 更新seen_tokens
        self.kv_seq_len = key_states.shape[2]
        past_key_value._seen_tokens = self.kv_seq_len

    # 递增KeyFormer迭代计数器
    self.keyformer_itr_count += 1

    return attn_output, None, past_key_value
