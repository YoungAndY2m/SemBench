"""
============================================================================
教学注释 (Annotation Pass) — SemBench L1 Code\* cars Q1 (text complaint filter)
============================================================================

cars Q1: 找出 "出过车祸" 的所有不同 car_id.
Pipeline: scan text_complaints → sem_filter (LLM 看 summary 字段判是否描述车祸)
          → project car_id → distinct.
特点: 纯 text-modal (depends_on=['summary']), 无 image / audio. 是 cars 10 个
query 里最简单的一个.

签名: run(pz_config, data_dir, scale_factor=157376)
- pz_config: QueryProcessorConfig 实例 (wrapper 传)
- data_dir: 数据集根目录
- scale_factor: cars 默认数据集规模 157376 行

注释说明: 本注释 pass 只增加 comment, 不改任何原始代码 (CLAUDE.md §5.5 §D 规则).
"""

import os
import pandas as pd
import palimpzest as pz


def run(pz_config, data_dir: str, scale_factor: int = 157376):
    # Load data
    complaints_text = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/text_complaints_data_{scale_factor}.csv"))
    complaints_text = pz.MemoryDataset(id="complaints", vals=complaints_text)

    # Filter data
    complaints_text = complaints_text.sem_filter('You are be given a textual complaint entailing that the car was in a crash/accident/collision. Complaint:', depends_on=['summary'])
    complaints_text = complaints_text.project(['car_id'])
    complaints_text = complaints_text.distinct(distinct_cols=['car_id'])
    
    output = complaints_text.run(pz_config)
    
    return output

