# flake8: noqa
# yapf: disable
from typing import Dict, List, Optional, Union
import inspect
import torch
from mmengine.device import is_npu_available
import transformers
from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
# 修改为
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    OffloadedCache,
    QuantizedCacheConfig,
    StaticCache,
)
PromptType = Union[PromptList, str]
import logging
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

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
from torch import nn
import torch.utils.checkpoint
from transformers.models.llama.modeling_llama import FlashAttentionKwargs
import torch.nn.functional as F
from typing_extensions import Unpack
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
from typing import Callable, List, Optional, Tuple, Union
from transformers import LlamaConfig

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

def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)


    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    # print("=====================================eager_attention_forward")

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float16).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # This is before the transpose
    seq_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=module.is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        target_dtype=target_dtype,
        **kwargs,
    )

    return attn_output, None

def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    def causal_mod(score, b, h, q_idx, kv_idx):
        if softcap is not None:
            score = softcap * torch.tanh(score / softcap)
        if causal_mask is not None:
            score = score + causal_mask[b][0][q_idx][kv_idx]
        if head_mask is not None:
            score = score + head_mask[b][h][0][0]
        return score

    attn_output, attention_weights = flex_attention(
        query,
        key,
        value,
        score_mod=causal_mod,
        enable_gqa=True,
        scale=scaling,
        # Last time checked on PyTorch == 2.5.1: Flex Attention always computes the lse regardless.
        # For simplification, we thus always return it as no additional computations are introduced.
        return_lse=True,
    )
    # lse is returned in float32
    attention_weights = attention_weights.to(value.dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attention_weights

def pure_scaled_dot_product_attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    attn_mask: Optional[torch.Tensor] = None, 
    dropout_p: float = 0.0, 
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:

    batch_size, num_heads, seq_len_q, head_dim = query.shape
    _, _, seq_len_k, _ = key.shape
    
   
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=query.device, dtype=torch.bool),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
    
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        else:
            attn_scores = attn_scores + attn_mask 

    attn_weights = F.softmax(attn_scores, dim=-1)
    
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    attn_output = torch.matmul(attn_weights, value)
    
    return attn_output

def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # print("=====================================sdpa_attention_forward")  
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    if is_causal is None:
        is_causal = causal_mask is None and query.shape[2] > 1

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # 纯PyTorch实现的scaled_dot_product_attention
    attn_output = pure_scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None

def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
        logger.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


ALL_ATTENTION_FUNCTIONS: Dict[str, Dict[str, Callable]] = {}

ALL_ATTENTION_FUNCTIONS.update(
    {
        "flash_attention_2": flash_attention_forward,
        "flex_attention": flex_attention_forward,
        "sdpa": sdpa_attention_forward,
    }
)
_torch_version = "N/A"
_torch_available = False
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_TF = os.environ.get("USE_TF", "AUTO").upper()

if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available, _torch_version = _is_package_available("torch", return_version=True)
else:
    logger.info("Disabling PyTorch because USE_TF is set")
    _torch_available = False

def is_torch_available():
    return _torch_available

def is_torchdynamo_compiling():
    if not is_torch_available():
        return False

    # Importing torch._dynamo causes issues with PyTorch profiler (https://github.com/pytorch/pytorch/issues/130622)
    # hence rather relying on `torch.compiler.is_compiling()` when possible (torch>=2.3)
    try:
        import torch

        return torch.compiler.is_compiling()
    except Exception:
        try:
            import torch._dynamo as dynamo  # noqa: F401

            return dynamo.is_compiling()
        except Exception:
            return False

def prepare_inputs_for_generation_llama(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is None: # [SnapKV]
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            # cache_length = past_length = past_key_values[0][0].shape[2]
            # max_cache_length = None
            cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = None
        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

def prepare_inputs_for_generation(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[Cache] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    
    # 直接输出past_key_values
    if past_key_values is None:  # [SnapKV]
        for layer in self.model.layers:
            print("layer.self_attn.kv_seq_len = 0")
            layer.self_attn.kv_seq_len = 0
    # 2. 有缓存但长度为0的情况
    elif isinstance(past_key_values, Cache) and past_key_values.get_seq_length() == 0:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
    # 1. Handle BC:
    model_inputs = {}
    # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
    if self._supports_cache_class:
        model_inputs["cache_position"] = cache_position
    elif cache_position is None:
        past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if past_key_values is not None:
        model_inputs["past_key_values"] = past_key_values
        if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
            inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
        elif (
            inputs_embeds is not None  # Exception 1
            or (is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1])  # Exception 3
        ):
            input_ids = input_ids[:, -cache_position.shape[0] :]
        elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
            input_ids = input_ids[:, cache_position]

    # 3. Prepare base model inputs
    input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step for every prompt.
    if not self.config.is_encoder_decoder:
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs[input_ids_key] = None
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            # `clone` calls in this function ensure a consistent stride. See #32227
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
            model_inputs["inputs_embeds"] = None
    else:
        model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

    # 4. Create missing `position_ids` on the fly
    attention_mask = (
        kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
    )
    attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
    position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
    if (
        attention_mask is not None
        and kwargs.get(position_ids_key) is None
        and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
    ):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        kwargs[position_ids_key] = position_ids  # placed in kwargs for further processing (see below)

    # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
    for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
        model_input = kwargs.get(model_input_name)
        if model_input is not None:
            if past_key_values is not None:
                current_input_length = (
                    model_inputs["inputs_embeds"].shape[1]
                    if model_inputs.get("inputs_embeds") is not None
                    else model_inputs[input_ids_key].shape[1]
                )
                model_input = model_input[:, -current_input_length:]
                model_input = model_input.clone(memory_format=torch.contiguous_format)
            model_inputs[model_input_name] = model_input

    # 6. Create 4D attention mask is we are using a `StaticCache` (important for performant compiled forward pass)
    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            device = model_inputs["inputs_embeds"].device
        else:
            batch_size, sequence_length = model_inputs[input_ids_key].shape
            device = model_inputs[input_ids_key].device

        # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
        # the 4D causal mask exists, it should be present in the base model (XXXModel class).
        base_model = getattr(self, self.base_model_prefix, None)
        if base_model is None:
            causal_mask_creation_function = getattr(
                self, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
        else:
            causal_mask_creation_function = getattr(
                base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
        if causal_mask_creation_function is None:
            logger.warning_once(
                f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                "writing code, see Llama for an example implementation. If you're a user, please report this "
                "issue on GitHub."
            )
        else:
            attention_mask = causal_mask_creation_function(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )
    if attention_mask is not None:
        model_inputs[attention_mask_key] = attention_mask

    # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
    for key, value in kwargs.items():
        if key not in model_inputs:
            model_inputs[key] = value

    # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
    model_inputs.pop("labels", None)
    return model_inputs

def prepare_inputs_for_generation_llama_modify(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    print("============ prepare_inputs_for_generation_llama_modify ========")
    # 直接输出past_key_values
    if past_key_values is None:  # [SnapKV]
        for layer in self.model.layers:
            print("layer.self_attn.kv_seq_len = 0")
            layer.self_attn.kv_seq_len = 0
    # 2. 有缓存但长度为0的情况
    elif isinstance(past_key_values, Cache) and past_key_values.get_seq_length() == 0:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0

    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens if hasattr(past_key_values, 'seen_tokens') else cache_length
            # 检查是否有 get_max_length 方法
            max_cache_length = past_key_values.get_max_length() if hasattr(past_key_values, 'get_max_length') else None
        else:
            cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = None
        
        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        print("初始化snapkv")
        print(f"SnapKVCluster 实例创建成功，配置参数：")
        print(f"- window_size: {window_size}")
        print(f"- max_capacity_prompt: {max_capacity_prompt}")
        print(f"- kernel_size: {kernel_size}")
        print(f"- pooling: {pooling}")
        print("=== SnapKV 初始化完成 ===")

        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        # print("=================update_kv")
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            print("q_len < self.max_capacity_prompt")
            return key_states, value_states
        else:
            # print("=================update_kv")
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states


def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        print("=== SnapKV 初始化开始 ===")
        
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 2
            print(f"设置默认 window_size: {self.config.window_size}")
        else:
            print(f"使用已有 window_size: {self.config.window_size}")
            
        if not hasattr(self.config, 'max_capacity_prompt'):
            # 默认是 2048
            self.config.max_capacity_prompt = 8
            print(f"设置默认 max_capacity_prompt: {self.config.max_capacity_prompt}")
        else:
            print(f"使用已有 max_capacity_prompt: {self.config.max_capacity_prompt}")
            
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
            print(f"设置默认 kernel_size: {self.config.kernel_size}")
        else:
            print(f"使用已有 kernel_size: {self.config.kernel_size}")
            
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
            print(f"设置默认 pooling: {self.config.pooling}")
        else:
            print(f"使用已有 pooling: {self.config.pooling}")
            
        print("创建 SnapKVCluster 实例...")
        self.kv_cluster = SnapKVCluster(
            window_size = self.config.window_size,
            max_capacity_prompt = self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling
        )


class SparseLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper with H2O sparse attention and SnapKV"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True


        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 初始化SnapKV
        init_snapkv(self)
        
        # 打印当前阶段信息
        # stage = "Prefill" if past_key_value is None else "Decode"
        # print(f"===== Stage: {stage} | Layer: {self.layer_idx} =====")

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
            attention_mask = kwargs.pop("padding_mask")

        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # print(f"Initial key_states shape: {key_states.shape} | Layer: {self.layer_idx}")

        # 计算序列长度
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            
            if hasattr(self, "kv_seq_len"):
                if self.kv_seq_len != 0:
                    kv_seq_len += self.kv_seq_len
                else:
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # 应用RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # 重复KV states
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # print(f"After repeat_kv, key_states shape: {key_states.shape} | Layer: {self.layer_idx}")

        # SnapKV缓存处理
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if key_states.shape[-2] == kv_seq_len:
                # print(f"===== SnapKV Compression - Layer: {self.layer_idx} =====")
                self.kv_seq_len = kv_seq_len
                # print(f"Before compression, key_states shape: {key_states.shape} | Layer: {self.layer_idx}")
                # print(f"Before compression, value_states shape: {value_states.shape} | Layer: {self.layer_idx}")
                
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                    key_states, 
                    query_states, 
                    value_states, 
                    attention_mask, 
                    self.num_key_value_groups
                )

                # print(f"After compression, key_states_compress shape: {key_states_compress.shape} | Layer: {self.layer_idx}")
                # print(f"After compression, value_states_compress shape: {value_states_compress.shape} | Layer: {self.layer_idx}")
                past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            else:
                # print(f"===== Cache Update - Layer: {self.layer_idx} =====")
                self.kv_seq_len += q_len
                old_key_shape = key_states.shape
                old_value_shape = value_states.shape
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                # print(f"Before cache update, key_states shape: {old_key_shape} | Layer: {self.layer_idx}")
                # print(f"After cache update, key_states shape: {key_states.shape} | Layer: {self.layer_idx}")
                # print(f"Before cache update, value_states shape: {old_value_shape} | Layer: {self.layer_idx}")
                # print(f"After cache update, value_states shape: {value_states.shape} | Layer: {self.layer_idx}")

        # 注意力计算
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # print(f"Final key_states shape before attention: {key_states.shape} | Layer: {self.layer_idx}")

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        # print(f"===== End of Layer: {self.layer_idx} =====\n")
        return attn_output, attn_weights

def forward1(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 初始化SnapKV
    init_snapkv(self)
        
    # 打印当前阶段信息
    # stage = "Prefill" if past_key_value is None else "Decode"
    # print(f"===== Stage: {stage} | Layer: {self.layer_idx} =====")

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
        attention_mask = kwargs.pop("padding_mask")

    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # print(f"Initial key_states shape: {key_states.shape} | Layer: {self.layer_idx}")

    # 计算序列长度
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
            
        if hasattr(self, "kv_seq_len"):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # 应用RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # 重复KV states
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
        
    # print(f"After repeat_kv, key_states shape: {key_states.shape} | Layer: {self.layer_idx}")

    # SnapKV缓存处理
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if key_states.shape[-2] == kv_seq_len:
            # print(f"===== SnapKV Compression - Layer: {self.layer_idx} =====")
            self.kv_seq_len = kv_seq_len
            # print(f"Before compression, key_states shape: {key_states.shape} | Layer: {self.layer_idx}")
            # print(f"Before compression, value_states shape: {value_states.shape} | Layer: {self.layer_idx}")
                
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                    key_states, 
                    query_states, 
                    value_states, 
                    attention_mask, 
                    self.num_key_value_groups
            )

            # print(f"After compression, key_states_compress shape: {key_states_compress.shape} | Layer: {self.layer_idx}")
            # print(f"After compression, value_states_compress shape: {value_states_compress.shape} | Layer: {self.layer_idx}")
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            # print(f"===== Cache Update - Layer: {self.layer_idx} =====")
            self.kv_seq_len += q_len
            old_key_shape = key_states.shape
            old_value_shape = value_states.shape
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # print(f"Before cache update, key_states shape: {old_key_shape} | Layer: {self.layer_idx}")
            # print(f"After cache update, key_states shape: {key_states.shape} | Layer: {self.layer_idx}")
            # print(f"Before cache update, value_states shape: {old_value_shape} | Layer: {self.layer_idx}")
            # print(f"After cache update, value_states shape: {value_states.shape} | Layer: {self.layer_idx}")

    # 注意力计算
    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # print(f"Final key_states shape before attention: {key_states.shape} | Layer: {self.layer_idx}")

    attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    # print(f"===== End of Layer: {self.layer_idx} =====\n")
    return attn_output, attn_weights




from importlib.metadata import version
import warnings
def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version

def replace_llama():
    transformers_version = check_version()
    version_list = ['4.49.0']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward1


# 定义替换函数
def replace_attention_with_layer_index(model):
    from transformers.models.llama.modeling_llama import LlamaAttention
    replace_llama()
    # 遍历模型的所有层
    # for layer_idx, layer in enumerate(model.model.layers):
    #     if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, LlamaAttention):
    #         custom_attn = SparseLlamaAttention(layer.self_attn.config, layer_idx=layer_idx)
    #         custom_attn.load_state_dict(layer.self_attn.state_dict(), strict=False)
    #         # 替换原始注意力层
    #         layer.self_attn = custom_attn
    #         print(f"Replaced attention in layer {layer_idx}")

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
class SnapKVLlamaAttentionConvert_1(BaseModel):

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

        self.other_kwargs = other_kwargs

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
        # =================== 替换snapkv ===================
        # replace_attention_with_layer_index(model = self.model)
        replace_attention_with_layer_index(self.model)
        self.model = self.model.half().cuda()
        print(self.model)
        # =================== 替换snapkv ===============================  


        # =================== 对推理进行监视 ===================
        from opencompass.models.profile_utils.timing_utils import global_monitor
        if not hasattr(global_monitor, '_hooks_registered'):
            global_monitor.register_hooks(self.model)
            global_monitor._hooks_registered = True
        # =================== 对推理进行监视 ===================




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
