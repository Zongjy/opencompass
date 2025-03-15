# configs/llama31_8b.py
from mmengine.config import read_base

with read_base():
    # 直接从预设数据集配置中读取需要的数据集配置
    from .datasets.longbenchv2.longbenchv2_gen import LongBenchv2_datasets  # 修改变量名以匹配数据集

# 将需要评测的数据集拼接成 datasets 字段
datasets = [*LongBenchv2_datasets]  # 使用更贴切的变量名

from opencompass.models import TurboMindModel

models = [
    dict(
        type=TurboMindModel,
        abbr='llama-3.1-8b-turbomind',
        path='meta-llama/Meta-Llama-3.1-8B',
        engine_config=dict(session_len=7168, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=1024),
        max_seq_len=7168,
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]