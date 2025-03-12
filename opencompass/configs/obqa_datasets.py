from mmengine.config import read_base

with read_base():
    from .datasets.commonsenseqa.commonsenseqa_gen_b57518 import commonsenseqa_datasets  # noqa: F401, F403
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets  # 修改变量名以匹配数据集
    from .datasets.longbench.longbenchmulti_news.longbench_multi_news_gen import LongBench_multi_news_datasets
    from .datasets.longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen import LongBench_multifieldqa_en_datasets
    from .datasets.longbench.longbenchvcsum.longbench_vcsum_gen import LongBench_vcsum_datasets

# 将需要评测的数据集拼接成 datasets 字段
datasets = [*gsm8k_datasets]  # 使用更贴切的变量名

from opencompass.models.sparse_attention.snap_kv import SnapKVLlamaAttentionConvert_1
from opencompass.models.sparse_attention.h2o import H2OLlamaAttentionConvert_1
from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=SnapKVLlamaAttentionConvert_1,
        abbr='llama-3_1-8b-instruct-turbomind',
        path='meta-llama/Meta-Llama-3.1-8B-Instruct',
        engine_config=dict(max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=4096),
        max_seq_len=16384,
        max_out_len=32,
        batch_size=1,
        run_cfg=dict(num_gpus=1,
                     heavy_ratio=0.5,
                     ),
        # modify here
        heavy_ratio=0.6,
        recent_ratio=0.6,
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        
    )
]