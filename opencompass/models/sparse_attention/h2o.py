# flake8: noqa
# yapf: disable
import argparse
from argparse import ArgumentParser
from typing import Dict, List, Optional, Union

import torch
from mmengine.device import is_npu_available

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]
import math
import pdb
import types
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          Cache, GenerationConfig, LlamaConfig)
# from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      LlamaForCausalLM,
                                                      LlamaRotaryEmbedding,
                                                      apply_rotary_pos_emb,
                                                      rotate_half)
from typing_extensions import Unpack

from opencompass.models import BaseModel
from opencompass.models.huggingface import HuggingFaceCausalLM
from opencompass.utils import get_logger

from .cake.utils import CompressConfig
from .monkeypatch import replace_llama, replace_mistral


def add_quest_args(parser: ArgumentParser):
    parser.add_argument('--dynamic-linear', action='store_true')
    parser.add_argument('--dynamic-ntk', type=float)
    parser.add_argument('--dynamic-part-ntk', action='store_true')
    parser.add_argument('--dynamic-yarn', action='store_true')
    parser.add_argument('--ntk', type=float)
    parser.add_argument('--part-ntk', type=float)
    parser.add_argument('--linear', type=float)
    parser.add_argument('--yarn', type=float)
    parser.add_argument('--rerope', type=float)
    parser.add_argument('--factor', type=float)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--finetuned', action='store_true')
    parser.add_argument('--gpt-neox-max-length', type=int)
    parser.add_argument('--adapter', type=str)
    parser.add_argument('--max-position-embeddings', type=int)
    parser.add_argument('--original-max-position-embeddings', type=int)
    parser.add_argument('--sliding-window-attention', type=int)
    parser.add_argument('--custom-model', action='store_true')
    parser.add_argument('--custom-model-together', action='store_true')
    parser.add_argument('--custom-model-mistral', action='store_true')
    parser.add_argument('--no-use-cache', action='store_true')
    return parser

# 定义替换函数
def replace_attention_with_layer_index(method = None,path = None):
    replace_llama(method)

@MODELS.register_module()
class H2OLlamaAttentionConvert_1(HuggingFaceCausalLM):
    def __init__(self, *args, **kwargs):
        self.past_key_values = None
        super().__init__(*args, **kwargs)

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):

        self._set_model_kwargs_torch_dtype(model_kwargs)
        method = 'sparq'
        # replace_attention_with_layer_index(method)
        if method ==  'flexprefill':
            model_kwargs['torch_dtype'] = torch.bfloat16
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        elif method == 'cake':
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
            compress_config = CompressConfig(True, True)
            gamma = 200
            compress_config.cache_size = 256
            compress_config.window_size = 32
            tau1 = 1.6
            tau2 = 0.4
            hyper = [tau1,tau2,gamma]
            compress_config.hyper = hyper

            config = AutoConfig.from_pretrained(path)
            if hasattr(config, 'num_hidden_layers'):
                layers = config.num_hidden_layers
            for i in range(layers):
                self.model.model.layers[i].self_attn.config.key_size = [compress_config.cache_size - compress_config.window_size]*layers
                self.model.model.layers[i].self_attn.config.window_size = [compress_config.window_size]*layers
                self.model.model.layers[i].self_attn.config.prefill = [True]*layers
                self.model.model.layers[i].self_attn.config.decoding_evict = [None]*layers
                self.model.model.layers[i].self_attn.config.tau1 = compress_config.hyper[0]
                self.model.model.layers[i].self_attn.config.tau2 = compress_config.hyper[1]
                self.model.model.layers[i].self_attn.config.gamma = compress_config.hyper[2]
                from .cake.cake_cache import CakeprefillKVCache
                self.model.model.layers[i].self_attn.config.prefill_cake_evict = [CakeprefillKVCache(
                        cache_size=compress_config.cache_size,
                        window_size=compress_config.window_size,
                        k_seq_dim=2,
                        v_seq_dim=2,
                        num_heads=self.model.model.layers[i].self_attn.num_heads,
                        num_layers=layers,
                        use_cascading=compress_config.cascading
                    )]*layers

        elif method == 'quest':
            parser = argparse.ArgumentParser()
            parser.add_argument('-m', '--model', action='append', nargs='+')
            parser.add_argument('--fixed-length', type=int)
            parser.add_argument('--max-tokens', type=int, default=8192)
            parser.add_argument('--min-tokens', type=int, default=256)
            parser.add_argument('--tokens-step', type=int)
            parser.add_argument('--length-step', type=int, default=128)
            parser.add_argument('--iterations', type=int, default=20)
            parser.add_argument('--output-file', type=str)

            parser.add_argument('--quest', action='store_true', help='Enable quest attention')
            parser.add_argument('--token_budget', type=int, default=1024)
            parser.add_argument('--chunk_size', type=int, default=16)
            from .Quest.evaluation.quest_attention import \
                enable_quest_attention_eval
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
            args = parser.parse_args([])  # 传入空列表，强制使用默认值  # 或使用 parser.parse_args([]) 强制使用默认值

            enable_quest_attention_eval(self.model, args)

        elif method == 'tova':
            import transformers

            from .tova.convert_models.convert import enable_tova_caching
            from .tova.convert_models.llama_custom import \
                OLD_LlamaRotaryEmbedding
            from .tova.tova_cache import TOVACache

            # transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = OLD_LlamaRotaryEmbedding
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
            enable_tova_caching(self.model)

            multi_state_size = 512
            cache = TOVACache(multi_state_size)
            self.past_key_values = cache

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)

        elif method == 'qfilters':
            from .qfilters.q_cache import KNormCache, QFiltersCache
            model_kwargs['torch_dtype'] = torch.bfloat16
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
            cache = QFiltersCache(
                self.model,
                window_length=64,
                max_length=128,
                model_name = path,
                **model_kwargs,
                )
            self.past_key_values = cache

        else:
            # 其它方法
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)

        replace_attention_with_layer_index(method, path)
        print(self.model)
        print('Model kwargs:')
        print(model_kwargs)

        # =================== 对推理进行监视 ===================
        from opencompass.models.profile_utils.timing_utils import \
            global_monitor
        if not hasattr(global_monitor, '_hooks_registered'):
            global_monitor.register_hooks(self.model)
            global_monitor._hooks_registered = True
        # =================== 对推理进行监视 ===================
        # print(self.model)
        self._set_model_kwargs_torch_dtype(model_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False
