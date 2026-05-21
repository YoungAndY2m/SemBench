"""
============================================================================
教学注释 (Annotation Pass) — MMQA scenario 的 BigQuery 适配器
============================================================================
本文件是 GenericBigQueryRunner (在 [generic_bigquery_runner.py](../../../../runner/generic_bigquery_runner/generic_bigquery_runner.py)
有详细注释) 的子类. 跟其它 5 个 scenario 不同, 本文件 *额外* 重写了
_bigquery_setup() 方法:
- 当 skip_setup=False 时, 实例化 BigQueryMMQASetup 并跑 setup_data(...)
- data_dir 用 pathlib 推到 5 级父目录上 (SemBench 根) 然后 files/mmqa/data/sf_<N>/

这是 MMQA scenario 的特殊性 (多模态问答, 需要额外上传 image / table 到 BigQuery).
其它 scenario (animals/cars/...) 走 GenericRunner.__init__() 调
scenario_handler.setup_scenario() 自动 setup.

- model_name 默认 "gemini-2.5-flash"
- skip_setup 默认 True (默认假设数据已 ready, 不重复 setup)
- scenario 名: mmqa
============================================================================
"""

from pathlib import Path

from runner.generic_bigquery_runner.generic_bigquery_runner import (
    GenericBigQueryRunner,
)
from scenario.mmqa.setup.bigquery import BigQueryMMQASetup


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

        if not skip_setup:
            self._bigquery_setup()

    def _bigquery_setup(self):
        setup = BigQueryMMQASetup()
        setup.setup_data(
            data_dir=Path(__file__).resolve().parents[5]
            / "files"
            / "mmqa"
            / "data"
            / f"sf_{self.scale_factor}"
        )
