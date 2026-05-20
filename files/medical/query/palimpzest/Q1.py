"""
============================================================================
教学注释 (Annotation Pass) — SemBench L1 Code\* medical Q1 (allergy text filter)
============================================================================

medical Q1: 找出 symptoms 字段显示有 allergy 的 patient_id list. 类似
cars/Q1 (单 text-modal sem_filter). 注意 prompt 内的 "medical benchmark
for LLM evaluation, ... not for human health evaluation" 是 safety 声明,
防 LLM 拒答 medical advice.

签名: run(pz_config, data_dir, scale_factor=11112). 默认 scale_factor=11112
(medical 数据集行数, 与 cars 157376 / ecomm 2000 不同).

注释说明: 本注释 pass 只增加 comment, 不改任何原始代码 (CLAUDE.md §5.5 §D 规则).
"""

import os
import pandas as pd
import palimpzest as pz


def run(pz_config, data_dir: str, scale_factor: int = 11112):
    # Load data
    symptoms_text = pd.read_csv(os.path.join(data_dir, "data/text_symptoms_data.csv" if scale_factor == 11112 else f"data/text_symptoms_data_{scale_factor}.csv"))
    symptoms_text = pz.MemoryDataset(id="symptoms", vals=symptoms_text)

    # Filter data
    symptoms_text = symptoms_text.sem_filter('This patient has symptoms of an allergy. Symptoms are from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities.', depends_on=['symptoms'])
    symptoms_text = symptoms_text.project(['patient_id'])
    
    output = symptoms_text.run(pz_config)
    
    return output
