# configs/llama31_8b.py
from mmengine.config import read_base

with read_base():
    # 直接从预设数据集配置中读取需要的数据集配置
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets  # 修改变量名以匹配数据集

# 将需要评测的数据集拼接成 datasets 字段
datasets = [*gsm8k_datasets]  # 使用更贴切的变量名

from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3_1-8b-instruct-hf',
        path='meta-llama/Meta-Llama-3.1-8B-Instruct',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]
