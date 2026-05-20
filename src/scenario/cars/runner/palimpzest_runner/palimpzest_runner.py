"""
Palimpzest system runner implementation.

============================================================================
教学注释 (Annotation Pass) — SemBench L1 ADD: cars Palimpzest wrapper
============================================================================

cars wrapper: Code\* mode (用户 query 是 Python 文件), 135 LoC.
**注意**: 这个 wrapper 含 6 个 bug / 不一致 (详 [Palimpzest LOG.md
遗留 #2-#8](../../../../../../../AllSQPE/Palimpzest/LOG.md)):

1. **typo 2 处** ([:110](#L110)): `Model.GPT_4o_MINI_AUDIO_PREVIE` 少
   一个 W; `Model.GPT_5_MINIW` 多一个 W. AttributeError fallback path.
2. **硬编码 MaxQuality** ([:117-129](#L117)): 不读 PALIMPZEST_CONFIG_FILE
   env var, 直接 `policy=pz.MaxQuality()`. 覆盖 generic wrapper 行为.
3. **`raise "string"`** ([:114](#L114)): 不合法 Python — 应该是
   `raise ValueError("...")`. 严格 Python 报 TypeError.
4. **is_audio = `query_id in [2,5,6,7,9]`** ([:86](#L86)): magic numbers,
   新加 audio query 会忘同步.
5. **execution_strategy="PARALLEL"** ([:119,130](#L119)): 大写; config JSON
   都用 lowercase "parallel". 看 Palimpzest 0.8.2.sem_agg 是否 case-
   insensitive.
6. **reasoning_effort="medium"** hardcoded ([:124](#L124)) for gemini-2.5-pro
   分支, 但 config JSON 全是 null. cars 自己 hardcode 赢.

Per-query mixed-models 策略 (audio query):
- query_id ∈ [2,5,6,7,9] → `available_models=[GPT_4o_AUDIO_PREVIEW, GPT_5_MINI]`
  (Palimpzest 的 mixture_of_agents 隐式路径)
- 否则 → single text model

scale_factor=157376 (cars 默认数据集行数, 写死).

注释说明: 本注释 pass 只增加 comment, 不改任何原始代码 (CLAUDE.md §5.5 §D 规则).
"""

from pathlib import Path
import sys
import time
import types
import traceback
import palimpzest as pz
from palimpzest.constants import Model

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
        model_name: str = "gpt-4o-audio-preview",
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
                self.get_palimpzest_config(
                    query_id=query_id, model_name=self.model_name
                ),
                self.scenario_handler.get_data_dir(),
                self.scale_factor
            )
            execution_time = time.time() - start_time

            # Store results in metric
            metric.execution_time = execution_time
            metric.results = (
                results.to_df()
                if not isinstance(results, tuple)
                else results[0]
            )
            metric.status = "success"
            metric.money_cost = (
                results.execution_stats.total_execution_cost
                if not isinstance(results, tuple)
                else results[1]
            )

        except Exception as e:
            metric.status = "failed"
            metric.error = str(e)
            print(f"  Error in Q{query_id} execution: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

        return metric

    def get_palimpzest_config(
        self, query_id, model_name
    ) -> pz.QueryProcessorConfig:
        is_audio = True if query_id in [2, 5, 6, 7, 9] else False

        model = None
        if model_name == "gpt-4o-mini":
            model = (
                [Model.GPT_4o_MINI_AUDIO_PREVIEW, Model.GPT_4o_MINI]
                if is_audio
                else [Model.GPT_4o_MINI]
            )
        elif model_name == "gpt-4o":
            model = (
                [Model.GPT_4o_AUDIO_PREVIEW, Model.GPT_4o]
                if is_audio
                else [Model.GPT_4o]
            )
        elif model_name == "gpt-5-mini":
            model = [Model.GPT_5_MINI]
        elif model_name == "gemini-2.0-flash":
            model = [Model.GEMINI_2_0_FLASH]
        elif model_name == "gemini-2.5-flash":
            model = [Model.GEMINI_2_5_FLASH]
        elif model_name == "gemini-2.5-pro":
            model = [Model.GEMINI_2_5_PRO]
        elif model_name == "gpt-4o-mini-audio-preview":
            model = [Model.GPT_4o_MINI_AUDIO_PREVIE, Model.GPT_5_MINIW]
        elif model_name == "gpt-4o-audio-preview":
            model = [Model.GPT_4o_AUDIO_PREVIEW, Model.GPT_5_MINI]
        else:
            raise "Unsupported model for Palimpzest"
        if model_name == "gemini-2.5-pro":
            return pz.QueryProcessorConfig(
                # policy=pz.MinTime(),
                policy=pz.MaxQuality(),
                execution_strategy="PARALLEL",
                max_workers=self.concurrent_llm_worker,
                verbose=False,
                progress=True,
                available_models=[model],
                reasoning_effort="medium",  # can be None (default), "low", "medium", or "high"
            )
        else:
            return pz.QueryProcessorConfig(
                # policy=pz.MinTime(),
                policy=pz.MaxQuality(),
                execution_strategy="PARALLEL",
                max_workers=self.concurrent_llm_worker,
                verbose=False,
                progress=True,
                available_models=model,
            )
