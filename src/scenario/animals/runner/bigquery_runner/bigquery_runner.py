"""
Created on Aug 8, 2025

@author: Jiale Lao

Animals BigQuery system runner implementation.

============================================================================
教学注释 (Annotation Pass) — Animals scenario 的 BigQuery 适配器
============================================================================
本文件是 GenericBigQueryRunner (在 [generic_bigquery_runner.py](../../../../runner/generic_bigquery_runner/generic_bigquery_runner.py)
有详细注释) 的薄子类, 只做 scenario-specific 配置:
- model_name 默认 "gemini-2.5-flash" (中端)
- skip_setup 默认 True (跳过数据初始化, 假设 BigQuery 表已经 ready)
- scenario 名: animals
core logic (jinja 模板 / cost 回收 / 重试) 全在父类, 这里只是把构造参数传上去.
============================================================================
"""

import time
from typing import Dict, List, override
import pandas as pd
from pathlib import Path
import os
from google.cloud import bigquery

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_runner import GenericQueryMetric
from runner.generic_bigquery_runner.generic_bigquery_runner import (
    GenericBigQueryRunner,
)

from scenario.animals.setup.bigquery import BigQueryAnimalsSetup


class BigQueryRunner(GenericBigQueryRunner):
    def __init__(
        self,
        use_case,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = True,
    ):
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )
        # Note: BigQuery setup is handled by scenario_handler.setup_scenario()
        # in GenericRunner.__init__() when skip_setup=False
