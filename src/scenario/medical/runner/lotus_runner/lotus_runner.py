"""
Lotus system runner implementation.
"""

# ============================================================
# 教学注释 (L1 wrapper pass):
# ------------------------------------------------------------
# Medical scenario 的 Code* mode wrapper. 与 cars 类似但有 2 个关键差异:
#
#   1) 用 scenario_handler.get_query_text + types.ModuleType + exec() 加载
#      query, 而非 importlib (line 47-51). 这是 SemBench 早期的 Code* 加载
#      方式, scenario_handler 抽象层管文件路径; cars 用更现代的 importlib
#      directly. 两种实现混存, 是历史包袱.
#
#   2) override execute_query (line 35-80) 而非靠 GenericLotusRunner 的
#      默认. 与父类几乎一样但 raise 替代 silent error (line 71) —
#      medical 实验更 strict: query 失败要立即停, 不允许吞 error.
#
# 默认 model gemini-2.5-pro 而非 gemini-2.5-flash (line 23) — medical
# query 需要更深推理, paper §7 实验默认 pro.
#
# 与 cars 共有 _execute_q<i> stub: 通过父类 GenericRunner._discover_queries
# (line 31-33) 而非 GenericLotusRunner 的 regex 发现 (因为 medical 没显式
# 写 _execute_q method, query 都从 scenario_handler 动态拿).
# ============================================================
from pathlib import Path
import sys
import time
import types
import lotus
import traceback

from runner.generic_runner import GenericQueryMetric, GenericRunner

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_lotus_runner.generic_lotus_runner import GenericLotusRunner


class LotusRunner(GenericLotusRunner):
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-pro",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        super().__init__(
            use_case, scale_factor, model_name, concurrent_llm_worker
        )

    def _discover_queries(self):
        # Match default implementation from GenericRunner
        return GenericRunner._discover_queries(self)

    # ========================================================
    # execute_query: 覆盖父类, 用 exec() 加载 query
    # --------------------------------------------------------
    # 加载流程 (line 47-54):
    #   1) scenario_handler.get_query_text(qid, "lotus") → str (Q<i>.py 全文)
    #   2) types.ModuleType("Q<i>_module") → 创建空 module 对象
    #   3) exec(query_text, module.__dict__) → 执行 Q<i>.py 全文, 把 def 注入
    #      到 module 命名空间
    #   4) module.run(data_dir, scale_factor) → 调 Q<i>.py 内的 run() 函数
    #
    # ⚠ exec() 是动态代码执行, 有 security 风险 (但 SemBench query 是
    # repository 自己的可信代码, 不接受 user input, OK).
    #
    # 与 cars 的 importlib 路径相比:
    #   - exec() 更轻 (无 spec/module_from_spec/loader 三件套)
    #   - 但 query 内的 import / __name__ 等 dunder 行为略不同
    # ========================================================
    def execute_query(self, query_id: int) -> GenericQueryMetric:
        metric = GenericQueryMetric(query_id=query_id, status="pending")

        # Reset token stats before each query
        try:
            lotus.settings.lm.reset_stats()
        except Exception as e:
            print(f"  Warning: Could not reset stats: {e}")

        try:
            # The queries in LOTUS are Python files with a run() function.
            # Load its contents, create a module, invoke the run() function.
            query_text = self.scenario_handler.get_query_text(
                query_id, self.get_system_name()
            )
            query_module = types.ModuleType(f"Q{query_id}_module")
            exec(query_text, query_module.__dict__)

            start_time = time.time()
            results = query_module.run(self.scenario_handler.get_data_dir(), self.scale_factor)
            execution_time = time.time() - start_time

            # Store results in metric
            metric.execution_time = execution_time
            metric.results = results
            metric.status = "success"

            # Get token usage and cost
            self._update_token_usage(metric)

        except Exception as e:
            # Handle failure
            metric.status = "failed"
            metric.error = str(e)
            print(f"  Error in Q{query_id} execution: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

        finally:
            # Reset stats after storing
            try:
                lotus.settings.lm.reset_stats()
            except:
                pass

        return metric
