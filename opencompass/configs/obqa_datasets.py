from mmengine.config import read_base
import torch

with read_base():
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets  # 修改变量名以匹配数据集

# 将需要评测的数据集拼接成 datasets 字段
datasets = [*gsm8k_datasets]  


from opencompass.models.sparse_attention.snap_kv import SnapKVLlamaAttentionConvert_1
from opencompass.models.sparse_attention.Replace_LlamaAttentionConvert import Replace_LlamaAttentionConvert
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.models.sparse_attention.infllm_model import INFLLM_LlamaForCausalLM

models = [
    dict(
        type=Replace_LlamaAttentionConvert,
        abbr='llama-3_1-8b-instruct-turbomind',
        path='meta-llama/Meta-Llama-3.1-8B-Instruct',
        max_seq_len=16384,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        method = 'minference',
        arkvale_kwargs=dict(
                page_size=32,
                # page_budgets=None, # page_budgets=None means "full" (no eviction & recall)
                page_budgets=1024 // 8,
                page_topks=8,
                n_max_bytes=4 * (1 << 30),
                n_max_cpu_bytes=80 * (1 << 30)
        ),
        infllm_kwargs=dict(
                model_center=False,
                type='inf-llm',
                block_size=64,
                fattn=False,
                n_init=128,
                n_local=1024,
                topk=16,
                repr_topk=4,
                max_cached_block=32,
                exc_block_size=512,
                base=500000,
                distance_scale=1.0,
        ),
        cache_kwargs=dict(
                window_size=128,
                max_capacity_prompt=512,
        )


        # modify here
    )
]
