"""
Created on August 4, 2025

@author: Jiale Lao

Movie BigQuery system runner implementation.

============================================================================
教学注释 (Annotation Pass) — Movie scenario 的 BigQuery 适配器
============================================================================
本文件是 GenericBigQueryRunner (在 [generic_bigquery_runner.py](../../../../runner/generic_bigquery_runner/generic_bigquery_runner.py)
有详细注释) 的薄子类 (跟 animals_runner 几乎完全相同):
- model_name 默认 "gemini-2.5-flash" (中端)
- skip_setup 默认 True (跳过数据初始化, 假设 BigQuery 表已 ready)
- scenario 名: movie
core logic 全在父类; 这里只是把构造参数传上去 + 一句 inline comment 说明
setup 走 scenario_handler.setup_scenario() (不是这里直接 setup).
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

from scenario.movie.setup.bigquery import BigQueryMovieSetup


class BigQueryRunner(GenericBigQueryRunner):
    def __init__(
        self,
        use_case: str,
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
