from mmengine.config import read_base

with read_base():
    from .datasets.squad20.squad20_gen import squad20_datasets  # noqa: F401, F403

# 将需要评测的数据集拼接成 datasets 字段
datasets = [*squad20_datasets]  # 使用更贴切的变量名

# from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models import H2OLlamaAttentionConvert_1

models = [
    dict(
        type=H2OLlamaAttentionConvert_1,
        abbr='llama-3_1-8b-instruct-turbomind',
        path='meta-llama/Meta-Llama-3.1-8B-Instruct',
        engine_config=dict(max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=4096),
        max_seq_len=16384,
        max_out_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
]