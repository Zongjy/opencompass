import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]

# from .huggingface import HuggingFaceCausalLM
from opencompass.models.huggingface import HuggingFaceCausalLM
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from .infllm_utils import patch_hf, GreedySearch, patch_model_center

@MODELS.register_module()
class INFLLM_LlamaForCausalLM(HuggingFaceCausalLM):
    def __init__(self, *args, **kwargs):
        self.infllm_kwargs = kwargs.pop('infllm_kwargs')
        super().__init__(*args, **kwargs)

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        # from transformers import AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)

        if self.infllm_kwargs.model_center:
            import bmtrain as bmt
            print("===============infllm====================`")
            bmt.init_distributed(seed=233)
            from model_center.model import Llama, LlamaConfig
            model_config = LlamaConfig.from_pretrained(path,token = token)

            model_config.dtype = torch.bfloat16
            self.model = Llama(model_config,token = token)
            bmt.load(self.model, os.path.join(path, "pytorch_model.pt"), strict=False)
            self.model = patch_model_center(self.model, self.infllm_kwargs.type, **self.infllm_kwargs)
            
            # =================== 对推理进行监视 ===================
            from opencompass.models.profile_utils.timing_utils import \
                global_monitor
            if not hasattr(global_monitor, '_hooks_registered'):
                global_monitor.register_hooks(self.model)
                global_monitor._hooks_registered = True
            # =================== 对推理进行监视 ===================
        else:
            print("========== infllm =========")
            self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
            self.model = patch_hf(self.model, self.infllm_kwargs.type, **self.infllm_kwargs)

             # =================== 对推理进行监视 ===================
            from opencompass.models.profile_utils.timing_utils import \
                global_monitor
            if not hasattr(global_monitor, '_hooks_registered'):
                global_monitor.register_hooks(self.model)
                global_monitor._hooks_registered = True
            # =================== 对推理进行监视 ===================

        # self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        
        self.model.eval()
        self.model.generation_config.do_sample = False