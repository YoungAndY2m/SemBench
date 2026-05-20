"""
Palimpzest system runner implementation.
Placeholder required by the current structure of the benchmarking framework.

============================================================================
教学注释 (Annotation Pass) — SemBench L1 ADD: ecomm Palimpzest wrapper
============================================================================

本文件是 SemBench L1 ADD 文件之一 (无对应 L0 — Palimpzest 没这个文件).
最简的 Code\* wrapper: 70 LoC, 只 override execute_query 走 Code\* mode.

Code\* mode 工作流 (SemBench 术语):
- Code\* = "Palimpzest Code dialect"; 用户 query 是一段 Python 代码
  (在 files/ecomm/queries/dialects/palimpzest/q*.py 中, 每个文件 def run
  函数), 用 Palimpzest fluent API 写 pipeline.
- wrapper.execute_query(qid):
    1. 读 query_text (Python 源码)
    2. exec() 成 module
    3. 调 module.run(pz_config, data_dir) → DataRecordCollection
    4. 拿 .to_df() / .execution_stats.total_execution_cost 包成 metric

scale_factor 不传 (ecomm 用户 query 不需 scale_factor; cars/medical 用).

⚠ 与 LOTUS ecomm wrapper 同款路径不一致问题: query 在
`files/ecomm/queries/dialects/palimpzest/` 而非 `files/ecomm/query/palimpzest/`
(其它 scenario `cars/medical` 用后者). 这是 SemBench 历史遗留.

money_cost = exec_stats.total_execution_cost: Palimpzest 自己累加的 LLM
USD 总额. SemBench `GenericQueryMetric.money_cost` 字段标准化为 USD.

注释说明: 本注释 pass 只增加 comment, 不改任何原始代码 (CLAUDE.md §5.5 §D 规则).
"""

from pathlib import Path
import sys
import time
import types
import traceback

from runner.generic_runner import GenericQueryMetric, GenericRunner

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_palimpzest_runner.generic_palimpzest_runner import (
    GenericPalimpzestRunner,
)


class PalimpzestRunner(GenericPalimpzestRunner):
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        super().__init__(
            use_case, scale_factor, model_name, concurrent_llm_worker
        )

    def _discover_queries(self):
        # Match default implementation from GenericRunner
        return GenericRunner._discover_queries(self)

    def execute_query(self, query_id: int) -> GenericQueryMetric:
        metric = GenericQueryMetric(query_id=query_id, status="pending")

        try:
            # The queries in Palimpzeset are Python files with a run() function.
            # Load its contents, create a module, invoke the run() function.
            query_text = self.scenario_handler.get_query_text(
                query_id, self.get_system_name()
            )
            query_module = types.ModuleType(f"q{query_id}_module")
            exec(query_text, query_module.__dict__)

            start_time = time.time()
            results = query_module.run(
                self.palimpzest_config(), self.scenario_handler.get_data_dir()
            )
            execution_time = time.time() - start_time

            # Store results in metric
            metric.execution_time = execution_time
            metric.results = results.to_df().rename(
                columns={"product_id": "id"}
            )
            metric.status = "success"
            metric.money_cost = results.execution_stats.total_execution_cost

        except Exception as e:
            metric.status = "failed"
            metric.error = str(e)
            print(f"  Error in Q{query_id} execution: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

        return metric
