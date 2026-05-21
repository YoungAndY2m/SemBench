"""
Generic ThalamusDB runner base class

@author: Jiale Lao
"""

# =============================================================================
# 教学注释 pass — SemBench L1 wrapper for ThalamusDB (paper-original SQPE)
# =============================================================================
# 这个文件是 SemBench 标准 `GenericRunner` 接口的 ThalamusDB 实现. 把 L0
# 的 `tdb.execution.engine.ExecutionEngine.run(query, constraints)` 包成
# `execute_query(query_id) → GenericQueryMetric` 让 SemBench 评测框架可用.
#
# L0 vs L1 责任分工:
#   - L0 (AllSQPE/ThalamusDB/): ExecutionEngine, anytime AQP, 2^N defaults,
#     NL operators — 算法+ system 本体
#   - L1 (this file): SemBench 适配, scenario 数据加载, token → money cost 转换,
#     query dispatcher (SQL 文件 vs Python method "Hybrid mode")
#
# 关键设计:
#
# 1. Hybrid query dispatcher (paper-original SQPE 独家):
#    execute_query(query_id) 先尝试 `scenario_handler.get_query_text(qid, 'thalamusdb')`
#    → 读 `files/<scenario>/query/thalamusdb/Q<id>.sql` 文件
#    找不到 fallback `_discover_query_impl(query_id)` → 调 self._execute_q<id>() 方法
#    (子类 e.g. movie/animals/mmqa/cars/medical/ecomm 各自实现).
#    这种 SQL-or-Python 混合方式 SemBench 里只有 ThalamusDB 用 (LOTUS 是 Code+Code*,
#    FlockMTL 纯 SQL, Palimpzest 纯 Python).
#
# 2. ⚠ AQP 默认关闭 (LOG_STRUCTURE.md §10.3 bug):
#       Constraints(max_calls=1e11, max_seconds=6000, max_tokens=1e22)
#    4 个 upper bound 都设到天文数字 + max_error 没设 (default 0) → 等同 exact mode.
#    ThalamusDB paper §6 主打 "anytime AQP", 但 SemBench 评测里 *从来没启用过*.
#    要做 PLAN P0 实验 (interval SE 对照) 必须 patch 这里, 把 4 个 constraint
#    暴露给 SemBench `--model` / config.
#
# 3. PRICING table — 把 LLM token usage 转 USD:
#    每个 model 的 (text_input, audio_input, output) per-million-token rate.
#    Audio token 单独算 (gpt-4o-audio-preview 输入 audio 更贵).
#    数字来源: OpenAI/Gemini 官方价格页面 2025 snapshot. 价格变动需 update.
#
# 4. model_name_to_file_name — 把 SemBench `--model gemini-2.5-flash` 映射到
#    config/system/thalamusdb/<file>.json 路径. 3 个 entry 写死, 加新 model
#    要改这个 dict + 加对应 json (LOTUS / Palimpzest 是 config 自动 lookup).
#
# 涉及的术语:
#   - GenericRunner / GenericQueryMetric — SemBench 标准 base class + dataclass
#     (其它 system 比如 LOTUS / Palimpzest 也都继承 GenericRunner)
#   - scenario_handler — SemBench 的 per-scenario data + query 加载工具
#   - Hybrid mode — SemBench 术语, 指 SQL 文件优先 + Python fallback 的 dispatcher
#   - max_error 默认 0 = exact mode = 跑到 UB == LB
#
# 调用方: SemBench framework 通过子类 (e.g. AnimalsThalamusDBRunner) 实例化 + 调
#         execute_query(qid). 子类一般 override _execute_q<id>() 方法.
# =============================================================================
import time
import pandas as pd
from typing import Dict, Any, List, Optional

import os
from pathlib import Path

# local codes version - set up sys.path to ensure consistent imports
# if you use the version from pip, please comment following codes
# import sys
# _tdb_path = os.path.join(os.path.dirname(__file__))
# if _tdb_path not in sys.path:
#     sys.path.insert(0, _tdb_path)
# end of local codes version

# pip install version
from tdb.data.relational import Database
from tdb.execution.constraints import Constraints
from tdb.execution.engine import ExecutionEngine
from tdb.queries.query import Query

import traceback
from ..generic_runner import GenericRunner, GenericQueryMetric


class GenericThalamusDBRunner(GenericRunner):
    """Base class for ThalamusDB system runners."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str,
        concurrent_llm_worker: int,
        db_path: str,
        skip_setup: bool = False,
    ):
        """
        Initialize the ThalamusDB runner.

        Args:
            use_case: The use case to run (e.g., 'detective')
            model_name: Name of the model to use
            concurrent_llm_worker: Number of concurrent LLM workers
            db_path: Path to the DuckDB database file
        """
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )

        # Initialize ThalamusDB components
        db_path = (
            db_path
            if db_path
            else os.path.join(
                self.scenario_handler.get_data_dir(), "thalamusdb.duckdb"
            )
        )
        self.db = Database(db_path)
        model_name_to_file_name = {
            "gemini-2.5-flash": "gemini_2.5flash",
            "gpt_5mini": "gpt_5mini",
            "gemini-2.5-pro": "gemini_2.5pro"
        }
        self.engine = ExecutionEngine(
            self.db,
            dop=20,
            model_config_path=f"{Path(__file__).resolve().parents[3]}/config/system/thalamusdb/{model_name_to_file_name[self.model_name]}.json",
        )
        self.constraints = Constraints(max_calls=100000000000, max_seconds=6000, max_tokens=10000000000000000000000)

    def get_system_name(self) -> str:
        """Return the name of the system."""
        return "thalamusdb"

    def execute_query(self, query_id: int) -> GenericQueryMetric:
        """
        Execute a specific query and return metric object with results.

        Args:
            query_id: ID of the query (e.g., 1 for Q1, 5 for Q5)

        Returns:
            GenericQueryMetric object containing results DataFrame and metrics
        """
        try:
            # Get the query implementation method
            # First try to find query files, if not found, fall back to _execute_q* methods
            query_fn = None
            if self.scenario_handler is not None:
                try:
                    query_text = self.scenario_handler.get_query_text(
                        query_id, self.get_system_name()
                    )
                    query_fn = lambda: self.execute_thalamusdb_query(query_text)
                except FileNotFoundError:
                    # Query file not found, try to use _execute_q* method
                    query_fn = self._discover_query_impl(query_id)
            else:
                query_fn = self._discover_query_impl(query_id)

            # Execute the query and track timing
            start_time = time.time()
            result = query_fn()
            execution_time = time.time() - start_time

            # Handle both DataFrame return (for backward compatibility) and dict with metrics
            if isinstance(result, pd.DataFrame):
                return GenericQueryMetric(
                    query_id=query_id,
                    status="success",
                    execution_time=execution_time,
                    results=result,
                    token_usage=0,
                    money_cost=0.0,
                )
            else:
                # Result is a dictionary with 'results', 'token_usage', 'money_cost'
                return GenericQueryMetric(
                    query_id=query_id,
                    status="success",
                    execution_time=execution_time,
                    results=result.get("results", pd.DataFrame()),
                    token_usage=result.get("token_usage", 0),
                    money_cost=result.get("money_cost", 0.0),
                )

        except Exception as e:
            print(f"Error executing query {query_id}: {e}")
            return GenericQueryMetric(
                query_id=query_id,
                status="failed",
                execution_time=0.0,
                error=str(e),
                results=self._get_empty_results_dataframe(query_id),
            )

    # -------------------------------------------------------------------------
    # execute_thalamusdb_query: 跑一条 ThalamusDB SQL → 返 (result_df, token, cost)
    # -------------------------------------------------------------------------
    # 流程:
    #   1. Query(self.db, sql_query) — 用 tdb.queries.Query 解析 (含 NL operator)
    #   2. 如果 query.semantic_predicates 非空 → engine.run() 走 anytime loop
    #      - 输出 (result_df, costs); costs.model2counters 是 dict
    #      - 遍历每个 model, 用 PRICING 算 cost_usd, 累加 token + cost
    #   3. 否则直接 self.db.execute() (pure SQL fast path) — 不走 anytime
    #
    # ⚠ "result_df, costs = engine.run(...)" 拆 tuple: ThalamusDB L0 返回
    #    (best_guess_df, TdbCounters). 这里只用了 best_guess, *丢掉了 LB/UB interval*
    #    → PLAN P0 interval SE 实验需要 patch 这里把 lower_bounds/upper_bounds 也存进 metric.
    #
    # ⚠ "Palimpzest might loose column names" 注释是 copy-paste 残留, 跟 ThalamusDB
    #    无关. ThalamusDB result 永远是 DataFrame, 不会走 isinstance(set) 分支.
    #
    # ⚠ non_audio_tokens = max(0, input_tokens - audio_tokens):
    #    防 input_tokens 没 split 模态时 (老模型 prompt_tokens_details 为 None) audio_tokens
    #    = 0 → 全算 text 价 (正确的回退).
    # -------------------------------------------------------------------------
    def execute_thalamusdb_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute a ThalamusDB SQL query and return results with metrics.

        Args:
            sql_query: ThalamusDB SQL query string

        Returns:
            Dictionary containing results, token_usage, and money_cost
        """
        try:
            query = Query(self.db, sql_query)

            if query.semantic_predicates:
                # Query has semantic predicates, use the execution engine
                result_df, costs = self.engine.run(query, self.constraints)

                # Convert result to DataFrame if it's not already
                if not isinstance(result_df, pd.DataFrame):
                    if hasattr(result_df, "df"):
                        result_df = result_df.df()
                    elif isinstance(result_df, set):
                        # Palimpzest might loose column names. We use 'id' on a best-effort.
                        result_df = pd.DataFrame.from_records(
                            result_df, columns=["id"]
                        )
                    else:
                        result_df = pd.DataFrame(result_df)

                # Pricing rules: (text_input, audio_input, output)
                PRICING = {
                    "gpt-4o": {"text": 2.5, "audio": 2.5, "output": 10.0},
                    "gpt-4o-mini": {"text": 0.15, "audio": 0.15, "output": 0.6},
                    "gpt-4o-audio-preview": {
                        "text": 2.5,
                        "audio": 2.5,
                        "output": 10.0,
                    },
                    "gpt-5": {"text": 1.25, "audio": 1.25, "output": 10.0},
                    "gpt-5-mini": {"text": 0.25, "audio": 0.25, "output": 2.0},
                    "gemini-2.0-flash": {
                        "text": 0.15,
                        "audio": 1.0,
                        "output": 0.6,
                    },
                    "gemini-2.5-flash": {
                        "text": 0.3,
                        "audio": 1.0,
                        "output": 2.5,
                    },
                    "gemini-2.5-flash-lite": {
                        "text": 0.1,
                        "audio": 0.3,
                        "output": 0.4,
                    },
                    "gemini-2.5-pro": {
                        "text": 1.25,
                        "audio": 1.25,
                        "output": 10.0,
                    },
                }

                token_usage_total = 0
                money_cost_total = 0.0
                per_model_costs = {}

                for model_name, counters in costs.model2counters.items():
                    input_tokens = getattr(counters, "input_tokens", 0)
                    audio_tokens = getattr(counters, "audio_input_tokens", 0)
                    output_tokens = getattr(counters, "output_tokens", 0)
                    non_audio_tokens = max(0, input_tokens - audio_tokens)
                    total_tokens = input_tokens + output_tokens

                    rates = PRICING.get(
                        model_name, {"text": 0.0, "audio": 0.0, "output": 0.0}
                    )
                    cost_usd = (
                        (non_audio_tokens * rates["text"] / 1_000_000.0)
                        + (audio_tokens * rates["audio"] / 1_000_000.0)
                        + (output_tokens * rates["output"] / 1_000_000.0)
                    )

                    per_model_costs[model_name] = {
                        "input_tokens": input_tokens,
                        "audio_tokens": audio_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "cost_usd": round(cost_usd, 6),
                    }

                    token_usage_total += total_tokens
                    money_cost_total += cost_usd

                return {
                    "results": result_df,
                    "token_usage": token_usage_total,
                    "money_cost": round(money_cost_total, 6),
                    "per_model": per_model_costs,
                }

            else:
                # Regular SQL query, execute directly on database
                result_df = self.db.execute(sql_query)

                if not isinstance(result_df, pd.DataFrame):
                    if hasattr(result_df, "df"):
                        result_df = result_df.df()
                    else:
                        result_df = pd.DataFrame(result_df)

                return {
                    "results": result_df,
                    "token_usage": 0,
                    "money_cost": 0.0,
                }

        except Exception as e:
            print(f"Error executing ThalamusDB query: {e}")
            traceback.print_exc()
            return {
                "results": pd.DataFrame(),
                "token_usage": 0,
                "money_cost": 0.0,
                "error": str(e),
            }
