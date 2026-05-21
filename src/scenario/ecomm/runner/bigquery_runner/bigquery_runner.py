"""
Snowflake system runner implementation.
Placeholder required by the current structure of the benchmarking framework.

============================================================================
教学注释 (Annotation Pass) — Ecomm scenario 的 BigQuery 适配器
============================================================================
★ 文档错位: 上面 docstring 写 "Snowflake" 是 copy-paste 失误, 本文件实际是
BigQuery 适配器 (从 import 和 super class GenericBigQueryRunner 可证). 注释 pass
不修原 docstring 文字 (CLAUDE.md §5.5 §D Rule 3), 只在新段标出.

本文件是 GenericBigQueryRunner (在 [generic_bigquery_runner.py](../../../../runner/generic_bigquery_runner/generic_bigquery_runner.py)
有详细注释) 的薄子类, 只做 scenario-specific 配置:
- model_name 默认 "gemini-2.5-flash" (中端)
- skip_setup 默认 False (会跑数据初始化)
- scenario 名: ecomm
core logic 全在父类.
============================================================================
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_bigquery_runner.generic_bigquery_runner import (
    GenericBigQueryRunner,
)


class BigQueryRunner(GenericBigQueryRunner):
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup=skip_setup,
        )
