# flake8: noqa
# yapf: disable
from typing import Dict, List, Optional, Union

import torch
from mmengine.device import is_npu_available

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from opencompass.models import BaseModel
from opencompass.utils import get_logger
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import math
from typing import Optional, Tuple
from transformers import Cache
import pdb
import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

# from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    LlamaForCausalLM,
)
import types

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class H2OKVCache_LayerWise:
    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

    def __call__(self, past_key_values, attn_score_cache):

        self._update_hh_score(attn_score_cache)

        if past_key_values is None:
            return (None,None)
        


        key_states = past_key_values.key_cache[self.layer_idx]  
        value_states = past_key_values.value_cache[self.layer_idx]


        # 获取序列长度
        seq_len = key_states.size(self.k_seq_dim)
        
        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = key_states.shape

        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values

        
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(key_states.device)
        mask = mask.scatter(-1, keep_idx, 1)

        # 调整 key_states 和 value_states 的索引
        key_states_flat = key_states.view(bsz * num_heads, seq_len, head_dim)
        value_states_flat = value_states.view(bsz * num_heads, seq_len, head_dim)
        mask = mask.unsqueeze(-1).expand(-1, -1, head_dim)  # [bsz * num_kv_heads, seq_len, head_dim]

        k_hh_recent = key_states_flat.squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = value_states_flat.squeeze()[mask].view(bsz, num_heads, -1, head_dim)

        self.hh_score= self.hh_score[mask].view(num_heads, self.cache_size)

        # 更新 DynamicCache
        past_key_values.key_cache[self.layer_idx] = k_hh_recent
        past_key_values.value_cache[self.layer_idx] = v_hh_recent

        return past_key_values

    def _update_hh_score(self, attn_score_cache):

        num_new_tokens = attn_score_cache.shape[2]
        print("attention_score_cache", attn_score_cache.size())
    
        # 假设 GQA 分组，例如 num_query_heads=32, num_kv_heads=8
        bsz, num_query_heads, seq_len, _ = attn_score_cache.shape
        num_kv_heads = 8  # 从 key_states 获取 num_kv_heads
        num_groups = num_query_heads // num_kv_heads  # 每个键/值头对应的查询头数量
    
        # 对注意力分数按组求和
        attn_score_cache = attn_score_cache.view(bsz, num_kv_heads, num_groups, seq_len, seq_len)
        attn_score_cache = attn_score_cache.sum(dim=2)  # 聚合组内查询头，形状 [bsz, num_kv_heads, seq_len, seq_len]
        attn_score_cache = attn_score_cache.sum(dim=-1)  # 求和得到 [bsz, num_kv_heads, seq_len]
        
        if self.hh_score is None:
            self.hh_score = attn_score_cache.view(bsz * num_kv_heads, seq_len)
            print(f"hh_score.shape: {self.hh_score.shape}")
        else:
            attn_score_cache = attn_score_cache.view(bsz * num_kv_heads, seq_len)
            attn_score_cache[:, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None

class Heavy_HitterLlamaAttention(nn.Module):
    """LLaMA 3 稀疏注意力层（最近 R + 重击 H 策略）"""

    def __init__(self, config, layer_idx=None, dtype=torch.float16):
        super().__init__()
        if layer_idx == 0:
            print("========= Using Sparse H2O-A Attention (Recent R + Heavy H) =========")

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias, dtype=dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, dtype=dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, dtype=dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias, dtype=dtype)

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # H2O-A 参数
        # self.recent_budget_ratio = config.recent_budget_ratio  
        # self.heavy_budget_ratio = config.heavy_budget_ratio  
        self.heavy_hitter_scores = None  # 存储 Token 累积注意力

        self.kv_cache = H2OKVCache_LayerWise(
            hh_size=4,
            recent_size=512,
            k_seq_dim=2,
            v_seq_dim=2,
        )

        self.kv_cache.layer_idx = layer_idx

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value:Optional[Tuple[torch.Tensor]] = None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # **Step 1: 计算 Q, K, V**
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # **Step 2: 旋转位置编码（RoPE）**
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # **Step 3: 处理 KV 缓存**
        if past_key_value is not None:
            # 更新 DynamicCache
            past_key_value.update(
                key_states=key_states,
                value_states=value_states,
                layer_idx=self.layer_idx,
            )
            # 从 DynamicCache 中提取当前层的 key 和 value
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
   
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float16).to(query_states.dtype)


        past_key_value = self.kv_cache(past_key_value, attn_weights.detach().clone())

        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # **Step 10: 计算最终注意力输出**
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class SparseLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: Optional[int] = None):
        if layer_idx == 0:
            print("=========using h20=======================")
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias,dtype=torch.float16)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias,dtype=torch.float16)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias,dtype=torch.float16)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias,dtype=torch.float16)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        self.heavy_budget_ratio = 0.6
        self.recent_budget_ratio = 0.6

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # ============================ h2o =============================================
        # attn_weights:[bsz, num_heads, q_len, k_len]
        heavy_budget = int(self.heavy_budget_ratio * attn_weights.shape[-1])
        recent_budget = int(self.recent_budget_ratio * attn_weights.shape[-1])
        # Heavy Hitter Mask (Based on global statistics)
        tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float16).to(attn_weights.dtype)
        
        tmp_sum = torch.sum(tmp_attn, dim=-2) 
        _, tmp_topk = tmp_sum.topk(k=heavy_budget, dim=-1)

        zeros = torch.zeros_like(tmp_sum, dtype=torch.bool)
        mask_bottom = zeros.scatter(-1, tmp_topk, True).unsqueeze(2)
        mask_bottom = mask_bottom.expand(mask_bottom.shape[0], mask_bottom.shape[1], attn_weights.shape[-2], mask_bottom.shape[-1])

        ones = torch.ones_like(attn_weights, dtype=torch.bool)
        ones = torch.triu(ones, diagonal=recent_budget)  # 前 recent_budget
        ones = torch.tril(ones, diagonal=-recent_budget)   # 后 recent_budget
        mask_bottom = torch.logical_or(mask_bottom, ones)
        # mask_bottom = ones
        attn_weights[~mask_bottom] = torch.finfo(attn_weights.dtype).min
        
        # ============================ h2o =============================================

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float16).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

import logging
# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

class H2OLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 先加载原始模型
        kwargs["torch_dtype"] = torch.float16
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        print(f"Loaded pretrained model from {pretrained_model_name_or_path}")
        
        # 获取模型的设备
        device = next(model.parameters()).device
        print(f"Model is on device: {device}")
        
        # 替换 self_attn 并同步设备
        num_layers = len(model.model.layers)
        print(f"Total number of layers: {num_layers}")
        for layer_idx in range(num_layers):
            old_attn = model.model.layers[layer_idx].self_attn.__class__.__name__
            model.model.layers[layer_idx].self_attn = SparseLlamaAttention(model.config, layer_idx)
            # 将新模块移动到与模型相同的设备
            model.model.layers[layer_idx].self_attn.to(device)
            new_attn = model.model.layers[layer_idx].self_attn.__class__.__name__
            print(f"Layer {layer_idx}: Replaced {old_attn} with {new_attn} on {device}")
        
        return model
    

# 定义替换函数
def replace_attention_with_layer_index(model):
    from transformers.models.llama.modeling_llama import LlamaAttention
    # 遍历模型的所有层
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
            # 创建自定义注意力层并传入层号
            custom_attn = SparseLlamaAttention(layer.self_attn.config, layer_idx=layer_idx)
            custom_attn.load_state_dict(layer.self_attn.state_dict(), strict=False)
            # 替换原始注意力层
            layer.self_attn = custom_attn
            print(f"Replaced attention in layer {layer_idx}")

def _get_stopping_criteria(stop_words, tokenizer, batch_size):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class MultiTokenEOSCriteria(StoppingCriteria):
        """Criteria to stop on the specified multi-token sequence."""

        def __init__(self, stop_words: List[str], tokenizer, batch_size: int):
            self.done_tracker = [False] * batch_size
            self.stop_words, self.max_sequence_id_len = [], 0
            for s in stop_words:
                self.stop_words.append(s)
                sequence_ids = tokenizer.encode(s, add_special_tokens=False)
                self.max_sequence_id_len = max(self.max_sequence_id_len, len(sequence_ids))
            self.tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            # compare the last len(stop) tokens
            lookback_ids_batch = input_ids[:, -self.max_sequence_id_len:]
            lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
            for i, done in enumerate(self.done_tracker):
                if done:
                    continue
                self.done_tracker[i] = any(s in lookback_tokens_batch[i] for s in self.stop_words)
            return False not in self.done_tracker

    c = MultiTokenEOSCriteria(stop_words, tokenizer, batch_size)
    return StoppingCriteriaList([c])

def _get_possible_max_seq_len(max_seq_len, path):
    if max_seq_len is not None:
        return max_seq_len

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    possible_keys = [
        'max_position_embeddings',
        'seq_length',
        'model_max_length',
    ]
    for k in possible_keys:
        if hasattr(config, k):
            return getattr(config, k)
    raise ValueError('max_seq_len is not provided and cannot be inferred from the model config.')


def _convert_chat_messages(inputs, merge_role=True, skip_empty_prompt=True):
    outputs = []
    for _input in inputs:
        messages = []
        if isinstance(_input, str):
            messages.append({'role': 'user', 'content': _input})
        else:
            for item in _input:
                if skip_empty_prompt and not item['prompt']:
                    continue
                role = {
                    'HUMAN': 'user',
                    'BOT': 'assistant',
                    'SYSTEM': 'system',
                }[item['role']]
                messages.append({'role': role, 'content': item['prompt']})

        if merge_role:
            merged_messages = []
            for item in messages:
                if merged_messages and merged_messages[-1]['role'] == item['role']:
                    merged_messages[-1]['content'] += '\n' + item['content']
                else:
                    merged_messages.append(item)
            messages = merged_messages

        outputs.append(messages)
    return outputs


def _format_with_fast_chat_template(inputs: List[str], name: str='vicuna'):
    try:
        from fastchat.model import get_conversation_template
    except ImportError:
        raise ModuleNotFoundError('fastchat not found. Please install with\npip install "fschat[model_worker,webui]"')

    outputs = []
    for _input in inputs:
        template = get_conversation_template(name)
        for item in _input:
            if item['role'] == 'user':
                template.append_message(template.roles[0], item['content'])
            elif item['role'] == 'assistant':
                template.append_message(template.roles[1], item['content'])
            elif item['role'] == 'system':
                continue
            else:
                raise ValueError(f"Unknown role {item['role']}")
        template.append_message(template.roles[1], None)
        outputs.append(template.get_prompt())
    return outputs


def _get_meta_template(meta_template):
    default_meta_template = dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            # XXX: all system roles are mapped to human in purpose
            dict(role='SYSTEM', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ]
    )
    return APITemplateParser(meta_template or default_meta_template)


def _set_model_kwargs_torch_dtype(model_kwargs):
    import torch
    if 'torch_dtype' not in model_kwargs:
        torch_dtype = torch.float16
    else:
        torch_dtype = {
            'torch.float16': torch.float16,
            'torch.bfloat16': torch.bfloat16,
            'torch.float': torch.float,
            'auto': 'auto',
            'None': None,
        }.get(model_kwargs['torch_dtype'])
    if torch_dtype is not None:
        model_kwargs['torch_dtype'] = torch_dtype
    return model_kwargs


@MODELS.register_module()
class H2OLlamaAttentionConvert(BaseModel):
    """Model wrapper for HuggingFace models designed for chat.

    Args:
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'mid' represents the part of input to
            truncate. Defaults to 'none'.
    """

    def __init__(self,
                 path: str,
                 model_kwargs: dict = dict(),
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 peft_kwargs: dict = dict(),
                 tokenizer_only: bool = False,
                 generation_kwargs: dict = dict(),
                 max_seq_len: Optional[int] = None,
                 meta_template: Optional[Dict] = None,
                 pad_token_id: Optional[int] = None,
                 fastchat_template: Optional[str] = None,
                 stop_words: Optional[str] = [],
                 mode: str = 'none',
                 **other_kwargs):

        self.logger = get_logger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.template_parser = _get_meta_template(meta_template)
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
        self._load_tokenizer(tokenizer_path or path, tokenizer_kwargs, pad_token_id)
        if not tokenizer_only:
            self._load_model(path=path, kwargs=model_kwargs, peft_path=peft_path, peft_kwargs=peft_kwargs)
        self.generation_kwargs = generation_kwargs
        self.fastchat_template = fastchat_template
        self.stop_words = list(set(stop_words + self._get_potential_stop_words(path)))
        assert mode in ['none', 'mid']
        self.mode = mode
        self.logger.info(f'using stop words: {self.stop_words}')

        for k, v in other_kwargs.items():
            if v is not None:
                self.logger.warning(f'Unused argument {k}={v}')

    def _load_tokenizer(self, path: Optional[str], kwargs: dict, pad_token_id: Optional[int] = None):
        from transformers import AutoTokenizer, GenerationConfig

        DEFAULT_TOKENIZER_KWARGS = dict(padding_side='left', truncation_side='left', trust_remote_code=True)
        tokenizer_kwargs = DEFAULT_TOKENIZER_KWARGS
        tokenizer_kwargs.update(kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)

        # A patch for some models without pad_token_id
        if pad_token_id is not None:
            if self.tokenizer.pad_token_id is None:
                self.logger.debug(f'Using {pad_token_id} as pad_token_id')
            elif self.tokenizer.pad_token_id != pad_token_id:
                self.logger.warning(f'pad_token_id is not consistent. Using {pad_token_id} as pad_token_id')
            self.tokenizer.pad_token_id = pad_token_id
            return
        if self.tokenizer.pad_token_id is not None:
            return
        self.logger.warning('pad_token_id is not set for the tokenizer.')
        generation_config = GenerationConfig.from_pretrained(path)
        if generation_config.pad_token_id is not None:
            self.logger.warning(f'Using {generation_config.pad_token_id} as pad_token_id.')
            self.tokenizer.pad_token_id = generation_config.pad_token_id
            return
        if self.tokenizer.eos_token_id is not None:
            self.logger.warning(f'Using eos_token_id {self.tokenizer.eos_token_id} as pad_token_id.')
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            return
        raise ValueError('pad_token_id is not set for this tokenizer. Please set `pad_token_id={PAD_TOKEN_ID}` in model_cfg.')

    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        from transformers import AutoModel, AutoModelForCausalLM

        DEFAULT_MODEL_KWARGS = dict(device_map='auto', trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS
        model_kwargs.update(kwargs)
        model_kwargs = _set_model_kwargs_torch_dtype(model_kwargs)
        self.logger.debug(f'using model_kwargs: {model_kwargs}')
        if is_npu_available():
            model_kwargs['device_map'] = 'npu'


        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        replace_attention_with_layer_index(self.model)
        # 将模型移动到 GPU
        self.model.to('cuda')
        # self.model = H2OLlamaForCausalLM.from_pretrained(path, **model_kwargs)

        # self.model = AutoModel.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            peft_kwargs['is_trainable'] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False


    def get_ppl_tokenwise(self, inputs: List[str], label: List[List[int]], mask_length: Optional[List[int]] = None) -> List[float]:
        """Get inference-ppl per token given a list of inputs and label.

        Args:
            inputs (List[str]): A list of strings.
            label (List[List[int]]): A list of list of label, each label is a tuple of (start, end, 1)
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        assert self.tokenizer.pad_token
        import torch
        import torch.nn.functional as F
        pad_token_id = self.tokenizer.pad_token_id
        messages = _convert_base_messages(inputs)

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len,
        )

        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'

        tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)

        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        outputs = self.model(**tokens)[0]

        batch_size, seq_len, vocab_size = outputs.shape
        shift_logits = outputs[:, :-1, :].contiguous().float()
        shift_labels = tokens['input_ids'][:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
            reduction='none').view(batch_size, seq_len - 1)
        lens = (tokens['input_ids'] != pad_token_id).sum(-1).cpu().numpy()

        if mask_length is not None:
            import numpy as np
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask
            lens -= np.array(mask_length)

        loss = loss.cpu().numpy()

        decode_messages = [[self.tokenizer.decode([input_id]) for input_id in token] for token in tokens['input_ids']]
        char_messages = [[ch for ch in message] for message in messages]

        # shifted to align label and loss
        for i in range(len(decode_messages)):
            decode_messages[i] = decode_messages[i][1:]

        aggregated_label_list = [[] for _ in range(len(decode_messages))]

        tag_list = [[] for _ in range(len(decode_messages))]

        for tmp_index, label_list in enumerate(label):
            for single_label in label_list:
                left = single_label[0]
                right = single_label[1]
                for i in range(left, right):
                    aggregated_label_list[tmp_index].append(i)


        def align_sequences(seq1, seq2, sep_len):
            """
            seq1: decoded sequence from token, one token may contain multiple characters
            seq2: original separate character sequence
            """
            i, j = 0, 0
            matched_pairs = []
            while i < len(seq1) and j < len(seq2):
                word = seq1[i]
                if len(word) == 0:
                    matched_pairs.append((word, []))
                    i += 1
                    continue

                if '\ufffd' in word:
                    for _ in range(sep_len):
                        matched_pairs.append((word, [j]))
                        i += 1
                    j += 1
                    continue

                char_sequence = ''
                while j < len(seq2) and (char_sequence != word):
                    char_sequence += seq2[j]
                    if char_sequence == word:
                        matched_pairs.append((word, [k for k in range(j - len(word) + 1, j+1)]))
                        j += 1
                        break
                    elif len(char_sequence) > len(word):
                        if word == char_sequence[-len(word):]:
                            matched_pairs.append((word, [k for k in range(j - len(word) + 1, j+1)]))
                            j += 1
                            break
                        else:
                            j += 1
                    else:
                        j += 1
                i += 1

            return matched_pairs



        if 'qwen' in self.path or 'Qwen' in self.path:
            sep_len = 2
        elif 'Llama-3' in self.path:
            sep_len = 2
        elif 'Yi' in self.path:
            sep_len = 3
        elif 'Llama-2' in self.path:
            sep_len = 3
        elif 'deepseek' in self.path:
            sep_len = 2
        else:
            sep_len = 3


        matched_pairs_list = [align_sequences(decode_messages[i], char_messages[i], sep_len) for i in range(len(decode_messages))]
        for match_index, matched_pairs in enumerate(matched_pairs_list):
            for i, (word, indices) in enumerate(matched_pairs):
                for j in indices:
                    if j in aggregated_label_list[match_index]:
                        tag_list[match_index].append(i)
                        break

        inference_loss_list = []
        token_len_list = []
        for i in range(len(loss)):
            inference_loss = 0
            token_len = 0
            for j in range(len(loss[i])):
                if j in tag_list[i]:

                    inference_loss += loss[i][j]
                    print(loss[i][j])
                    token_len += 1
            inference_loss_list.append(inference_loss)
            token_len_list.append(token_len)

        return inference_loss_list, token_len_list

    def _get_potential_stop_words(self, path: Optional[str]):
        from transformers import GenerationConfig
        potential_stop_words = []
        try:
            generation_config = GenerationConfig.from_pretrained(path)
        except:
            generation_config = None
        if generation_config and hasattr(generation_config, 'eos_token_id'):
            if isinstance(generation_config.eos_token_id, int):
                potential_stop_words.append(self.tokenizer.decode(generation_config.eos_token_id))
            else:
                assert isinstance(generation_config.eos_token_id, list)
                for token_id in generation_config.eos_token_id:
                    potential_stop_words.append(self.tokenizer.decode(token_id))
        if self.tokenizer.eos_token is not None:
            potential_stop_words.append(self.tokenizer.eos_token)
        potential_stop_words = list(set(potential_stop_words))
        potential_stop_words = [s for s in potential_stop_words if s]
        return potential_stop_words

    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = [],
                 **kwargs) -> List[str]:
        messages = _convert_chat_messages(inputs)
        batch_size = len(messages)

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len
        )
        if self.fastchat_template:
            messages = _format_with_fast_chat_template(messages, self.fastchat_template)
            tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)
        else:
            messages = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]
            tokenize_kwargs['add_special_tokens'] = False
            tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)

        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

        if self.mode == 'mid':
            # Reserve space for the tokens to be generated in the future.
            max_prompt_len = self.max_seq_len - max_out_len

            # Retain the first 0.5 * max_prompt_len tokens and the last 0.5 * max_prompt_len tokens, discarding the middle ones,
            # because the prompts' questions are usually at the beginning or the end.
            # To avoid the warning:
            # This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length.
            # Depending on the model, you may observe exceptions, performance degradation, or nothing at all.
            half_max_prompt_len = max_prompt_len // 2
            if half_max_prompt_len > 0 and tokens['input_ids'].shape[1] > max_prompt_len:
                for key in tokens.keys():
                    if tokens[key].shape[1] > max_prompt_len:
                        field_values = tokens[key]
                        tokens[key] = torch.cat(
                            (field_values[:, :half_max_prompt_len], field_values[:, -half_max_prompt_len:]), dim=1
                        )

        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update(kwargs)
        stopping_criteria = list(set(stopping_criteria + self.stop_words))
        if stopping_criteria:
            generation_kwargs['stopping_criteria'] = _get_stopping_criteria(stopping_criteria, self.tokenizer, batch_size)
        if max_out_len is not None:
            generation_kwargs['max_new_tokens'] = max_out_len
        if min_out_len is not None:
            generation_kwargs['min_new_tokens'] = min_out_len
        generation_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        self.logger.info('Generation Args of Huggingface: ')
        self.logger.info(generation_kwargs)

        # step-2: conduct model forward to generate output
        outputs = self.model.generate(**tokens, **generation_kwargs)
        
        outputs = outputs[:, tokens['input_ids'].shape[1]:]

        # step-3: decode the output
        decodeds = self.tokenizer.batch_decode(outputs)
        for stop in stopping_criteria:
            decodeds = [t.split(stop)[0] for t in decodeds]

        return decodeds

    def get_token_len(self, prompt: str) -> int:
        m = _convert_chat_messages([prompt])[0]
        t = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_dict=True)
        return len(t['input_ids'])

def  _convert_base_messages(inputs):
    outputs = []
    for _input in inputs:
        if isinstance(_input, str):
            outputs.append(_input)
        else:
            messages = []
            for item in _input:
                messages.append(item['prompt'])
            outputs.append(''.join(messages))
    return outputs
