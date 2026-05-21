"""
BigQuery runner implementation.

============================================================================
教学注释 (Annotation Pass) — Cars scenario 的 BigQuery 适配器
============================================================================
本文件是 GenericBigQueryRunner (在 [generic_bigquery_runner.py](../../../../runner/generic_bigquery_runner/generic_bigquery_runner.py)
有详细注释) 的薄子类, 只做 scenario-specific 配置:
- model_name 默认 "gemini-2.5-pro" (高端, cars 的 query 多需要 reasoning)
- skip_setup 默认 False (会跑数据初始化)
- scenario 名: cars
core logic (jinja 模板 / cost 回收 / 重试) 全在父类, 这里只是把构造参数传上去.
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
        model_name: str = "gemini-2.5-pro",
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
