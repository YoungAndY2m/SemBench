"""
============================================================================
教学注释 (Annotation Pass) — Ecomm scenario 的 evaluator
============================================================================
本文件是 GenericEvaluator (在 [src/evaluator/generic_evaluator.py](../../../evaluator/generic_evaluator.py)
有详细注释) 的薄子类. 是 6 个 scenario evaluator 中 *最瘦* 的, 因为 ecomm
scenario 的 ground truth + per-query 评估方法都已经在 EcommScenario handler
里 (见 [src/scenario/ecomm/ecomm_scenario.py](../ecomm_scenario.py)), 这里只是把
两个接口转接过去:
- `_get_ground_truth(qid)` → `scenario_handler.get_ground_truth(qid)`
- `_evaluate_single_query(qid, sys, gt)` → 用 `scenario_handler.get_accuracy_measure_for_query(qid)`
  拿到 measure type (retrieval / aggregation / single_accuracy), 再调
  `GenericEvaluator.compute_accuracy_score(...)` 计算分数

★ 注释说一句: docstring 里说 "Ideally, we would only have one intantiation of
the scenario handler in run.py" — 即理想情况 scenario_handler 应该在 run.py 入口
只 new 一次, 然后传给 runner + evaluator 共用; 当前 evaluator 自己又 new 一个
EcommScenario, 是 in-progress refactor 的遗留.

注释说明: 本注释 pass 只增加 comment, 不改任何原始代码.
============================================================================
"""

from pathlib import Path
from typing import Any, Dict
import sys
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from evaluator.generic_evaluator import (
    GenericEvaluator,
    QueryMetricRetrieval,
    QueryMetricAggregation,
    SingleAccuracyScore,
)

sys.path.append(str(Path(__file__).parent.parent))
from ecomm.ecomm_scenario import EcommScenario


class EcommEvaluator(GenericEvaluator):
    def __init__(self, use_case: str, scale_factor: int = None) -> None:
        super().__init__(use_case, scale_factor)
        # Ideally, we would only have one intantiation of the scenario handler in run.py
        self.scenario_handler = EcommScenario(scale_factor=scale_factor)

    def _load_domain_data(self) -> None:
        # Not needed because this is already done by the ecomm scenario handler
        pass

    def _get_ground_truth(self, query_id: int) -> pd.DataFrame:
        return self.scenario_handler.get_ground_truth(query_id)

    def _evaluate_single_query(
        self,
        query_id: int,
        system_results: pd.DataFrame,
        ground_truth: pd.DataFrame,
    ) -> "QueryMetricRetrieval | QueryMetricAggregation | SingleAccuracyScore":
        # The following code is very generic and could be used for any scenario
        eval_measure = self.scenario_handler.get_accuracy_measure_for_query(
            query_id
        )
        return GenericEvaluator.compute_accuracy_score(
            eval_measure, ground_truth, system_results
        )
