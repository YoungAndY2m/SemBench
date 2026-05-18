"""
Created on August 1, 2025

@author: Jiale Lao

Generic CAESURA runner for MMBench-System

============================================================================
SemBench/src/runner/generic_caesura_runner/generic_caesura_runner.py
SemBench L1 wrapper — 把 CAESURA L0 agent 适配到 SemBench `GenericRunner` 接口
============================================================================

教学注释 pass (L1 ADD) by Claude.
对应 [CAESURA LOG_STRUCTURE.md §10.1.1 L1 ADD](../../../../AllSQPE/CAESURA/LOG_STRUCTURE.md#1011-l1-add--新增文件l0-没有的)
和 [§10.2 L0 → L1 抽象增益总结](../../../../AllSQPE/CAESURA/LOG_STRUCTURE.md#102-l0--l1-的抽象增益一句话总结)。

----------------------------------------------------------------------------
这个文件干什么 (一句话)
----------------------------------------------------------------------------
L0 的 CAESURA agent 设计是"一次性 `agent.run(query)` 出 DataFrame", 没有 *evaluation framework*
所需的:
  - per-query reset (跨 query 复用 agent 实例)
  - token usage / cost tracking
  - schema-aware 期望结果验证 (SemBench evaluator 用)
  - per-scenario 数据 setup (load CSV / link / build index)

本文件就是这个适配层 — 继承 `GenericRunner` (SemBench 内统一 runner 接口),
对接 L0 `Caesura` agent, 暴露 SemBench 标准的 `execute_query(query_id)` API。

----------------------------------------------------------------------------
L0 → L1 抽象增益 (3 件事)
----------------------------------------------------------------------------
1. **Per-query state lifecycle**:
   L0 的 `agent.run(query)` 是 "用完即弃"; L1 通过 `Caesura.reset_for_new_query()` (L1 patch in main.py)
   让 agent 跨多 query 复用 — 重置 chat history / working set / token counter。

2. **Token / cost tracking**:
   L0 LLM 调用没暴露 usage; L1 在 `MyOpenAI._generate` (L1 patch in model.py) 截取
   `result.llm_output['token_usage']` 累加, 给 `_update_token_usage_and_cost` 算实际费用。

3. **GenericRunner 接口适配**:
   - `execute_query(query_id) → GenericQueryMetric` (而不是 L0 的 NL str)
   - `_discover_query_impl(query_id)` 用反射找 `_execute_q1` / `_execute_q2` ... 方法
   - per-scenario subclass 覆盖 `_setup_database_from_files()` 做数据清洗

----------------------------------------------------------------------------
零基础读者预备知识 (Python)
----------------------------------------------------------------------------
- `sys.path.insert(0, path)`:
  Python import 时按 sys.path 顺序找包; insert(0, ...) 把当前目录加到搜索最前面,
  确保本地 vendored `caesura/` 包 *优先* 于全局安装的 caesura (若有).

- `hasattr(self, attr_name)` + `getattr(self, attr_name)` (反射):
  动态查找属性 — `_discover_query_impl` 用它找子类是否实现了 `_execute_q5` 这种方法。

- `re.compile(r"_execute_q(\d+)$")`:
  预编译正则 (比 re.match 多次调用快); 捕获 `_execute_q123` 中的数字部分。

- `from ..generic_runner import ...`:
  双点 `..` 相对 import — 跳到上一层包 (`src/runner/`), 再进 `generic_runner` 模块。

- `super().__init__(use_case, model_name, scale_factor, ...)`:
  ⚠ 注意! GenericRunner.__init__ 的签名是 `(use_case, scale_factor, model_name, ...)`,
  但这里传的顺序是 `(use_case, model_name, scale_factor, ...)` —— 参数顺序错位!
  详 [LOG.md "遗留问题 #1"](../../../../AllSQPE/CAESURA/LOG.md):
  结果: self.scale_factor = "gpt-4-0613" (str), self.model_name = 2000 (int);
  self.data_path = files/movie/data/sf_gpt-4-0613/ → 路径不存在。
  历史 bug, 至今未修。

----------------------------------------------------------------------------
依赖关系
----------------------------------------------------------------------------
L0 vendored: `caesura/` (本目录下整包) — 通过 sys.path hack import
SemBench:    `GenericRunner` (上一层 `generic_runner.py`) — 通过 `..` 相对 import

per-scenario subclass: 在 SemBench/src/scenario/<use_case>/runner/caesura_runner/caesura_runner.py
继承 GenericCaesuraRunner; 覆盖 `_setup_database_from_files` + 实现 `_execute_q1..N` + `_get_empty_results_dataframe`。

详细教程见 [EXTENSION.md](EXTENSION.md)。
"""

import time
import pandas as pd
from typing import Dict, Any, List, Optional
import sys
import os
import re

# Set up sys.path to ensure consistent imports for local CAESURA code
# 关键 hack — 把本目录加到 sys.path 最前面, 让 vendored caesura/ 包优先于全局 caesura (若有)
# os.path.dirname(__file__) = generic_caesura_runner/ 本目录
_caesura_path = os.path.join(os.path.dirname(__file__))
if _caesura_path not in sys.path:
    sys.path.insert(0, _caesura_path)

# Import CAESURA components
# 这两个 import 必须在 sys.path 修改之后, 否则 Python 找不到 vendored caesura
from caesura.main import Caesura
from caesura.scenarios import get_database

import traceback
# 相对 import — 跳上一层到 src/runner/, 再进 generic_runner 模块
# GenericRunner 是 SemBench 全部 system runner 的统一基类; GenericQueryMetric 是结果数据类
from ..generic_runner import GenericRunner, GenericQueryMetric


class GenericCaesuraRunner(GenericRunner):
    """Generic CAESURA runner for MMBench-System."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str,
        concurrent_llm_worker: int,
        skip_setup: bool = False,
    ):
        """
        Initialize the CAESURA runner.

        Args:
            use_case: The use case to run (e.g., 'movie', 'detective', 'animals')
            model_name: Name of the model to use (e.g., 'gpt-4-0613', 'gpt-3.5-turbo-0613')
            concurrent_llm_worker: Number of concurrent LLM workers (not used by CAESURA)
            skip_setup: Whether to skip setup (inherited from GenericRunner)
        """
        # ⚠ BUG: 参数顺序错! GenericRunner.__init__ 期望 (use_case, scale_factor, model_name, ...)
        # 但这里传的是 (use_case, model_name, scale_factor, ...);
        # 结果: self.scale_factor = "gpt-4-0613" (str), self.model_name = 2000 (int)
        # 详 file docstring + LOG.md "遗留问题 #1"
        super().__init__(
            use_case,
            model_name,
            scale_factor,
            concurrent_llm_worker,
            skip_setup,
        )

        # Map common model names to CAESURA format
        # SemBench 的 model name 体系 (e.g. "gpt-4" / "gemini-2.5-flash") 与 CAESURA hardcode 的
        # 0613 snapshot 名不匹配 — _map_model_name 做 fallback 转换
        self.caesura_model = self._map_model_name(model_name)

        # Initialize CAESURA database using MMBench data files
        # 把 SemBench 标准 files/<use_case>/data/sf_<scale>/ 目录下的 CSV 加载成 CAESURA Database
        # ⚠ 因为参数顺序 bug, self.data_path 实际是 files/movie/data/sf_gpt-4-0613/ (不存在),
        # 这里 setup 几乎一定失败 — 但 LOG.md "遗留问题 #1" 提到 raw_results 里 Q*.csv 存在,
        # 说明 CAESURA 历史上确实跑了 — 路径可能在旧版本不同
        try:
            self.database = self._setup_database_from_files()
        except Exception as e:
            raise ValueError(
                f"Failed to setup CAESURA database for use case '{use_case}': {e}"
            ) from e

        # Initialize single CAESURA agent for reuse
        # 关键 L1 抽象: 单个 agent 实例跨 query 复用 (L0 没考虑过)
        # interactive=False — paper §6 默认 batch mode; 关掉 input() 阻塞 (Caesura.run 内的 user prompt 也会跳过)
        self.caesura_agent = Caesura(
            database=self.database,
            model_name=self.caesura_model,
            interactive=False,
        )

    def _map_model_name(self, model_name: str) -> str:
        """Map MMBench model names to CAESURA model names."""
        # CAESURA 只认 "gpt-3.5-turbo-0613" 和 "gpt-4-0613" 两个固定快照 (硬编码在 model.py);
        # SemBench 用户传 "gpt-4" 或现代 "gemini-2.5-flash" 等都得映射到 0613;
        # 无映射时 fallback 到 "gpt-4-0613" — 即任何输入 *默认* 落到 GPT-4 价格区间
        # 这意味着即使用户传 "gemini-2.5-flash", CAESURA 内部仍然会按 gpt-4-0613 的 rate limit / token budget 跑;
        # 但 OpenAI API key 没 0613 模型权限的话会直接 401, 见 LOG.md pitfall #5
        model_mapping = {
            "gpt-4": "gpt-4-0613",
            "gpt-3.5": "gpt-3.5-turbo-0613",
        }
        return model_mapping.get(model_name, "gpt-4-0613")  # Default to GPT-4

    def _setup_database_from_files(self):
        """Setup CAESURA database using MMBench data files."""
        # =====================================================================
        # 把 SemBench 标准目录结构下的 CSV 加载成 CAESURA Database
        # =====================================================================
        # 3 个 use case 各有不同 schema; movie 是 paper 复现最完整的 (含 link + index),
        # detective 和 animals 是 stub (只 load 表, 没 link 没 index)
        #
        # per-scenario subclass (e.g. movie/caesura_runner/caesura_runner.py) 会覆盖本方法
        # 加更细致的数据清洗 (drop 列 / dropna / 重排列序)。
        from caesura.database.database import Database

        db = Database()

        if self.use_case == "movie":
            # Load movie data from MMBench files
            # ⚠ 路径 hardcode "Movies_2000.csv" / "Reviews_2000.csv";
            # 实际 SemBench 当前 data 目录里只有 "Movies.csv" / "Reviews.csv"
            # 详 LOG.md "遗留问题 #2"
            movies_path = str(self.data_path / "Movies_2000.csv")
            reviews_path = str(self.data_path / "Reviews_2000.csv")

            # Add movies table
            db.add_tabular_table(
                "movies",
                movies_path,
                "a table that contains information about movies including titles, scores, ratings, genres, directors, and other metadata",
            )

            # Add reviews table
            # 注意! 是 add_text_table — 因为最后一列是 reviewText (长文本)
            # CAESURA 约定: text table 最后一列被标 TEXT 类型 (table.py:create_text_table)
            db.add_text_table(
                "reviews",
                reviews_path,
                "a table that contains movie reviews with metadata and review text content for sentiment analysis",
            )

            # Link reviews to movies by movie id
            # link("reviews", "movies", "id"): reviews.id ↔ movies.id 单向连接
            # 注意 link 是单向的; 这里只挂到 reviews.links, 不挂到 movies.links
            # Discovery DFS join 算法依赖单向边的有向图假设
            db.link("reviews", "movies", "id")

            # Build relevant values index for key categorical columns
            # n-gram 倒排索引 — Discovery Phase 用 query keyword 模糊匹配回真实列值
            # 选哪几列建索引是经验值; 一般是 *categorical + 用户在 query 常引用* 的列
            db.build_relevant_values_index(
                "movies", "genre", "director", "rating", "originalLanguage"
            )
            db.build_relevant_values_index(
                "reviews", "reviewState", "publicationName", "isTopCritic"
            )

        elif self.use_case == "detective":
            # =================================================================
            # ⚠ detective: stub 实现 — get_available_use_cases() 返回 detective 但实际:
            #   1. SemBench 的 detective scenario 目录下 *没有 caesura_runner/ subdir*
            #   2. 本分支只 load 表, 没建 link / index — CAESURA 跑不通完整 query
            # 详 LOG.md "遗留问题 #3" 已记录这是 dead reference
            # =================================================================
            # For detective case, we might need to handle DuckDB files differently
            # For now, let's implement basic CSV loading
            dmv_path = str(self.data_path / "dmv_table.csv")
            evidence_path = str(self.data_path / "evidence_table.csv")
            shop_cams_path = str(self.data_path / "shop_cams_table.csv")
            traffic_cams_path = str(self.data_path / "traffic_cams_table.csv")

            db.add_tabular_table("dmv", dmv_path, "DMV records table")
            db.add_tabular_table("evidence", evidence_path, "Evidence table")
            db.add_tabular_table(
                "shop_cams", shop_cams_path, "Shop camera footage table"
            )
            db.add_tabular_table(
                "traffic_cams",
                traffic_cams_path,
                "Traffic camera footage table",
            )

        elif self.use_case == "animals":
            # ⚠ animals: 也是 stub — 与 detective 同, 没有 per-scenario subclass 接入
            # Load animal data
            audio_path = str(self.data_path / "audio_data.csv")
            image_path = str(self.data_path / "image_data.csv")

            db.add_tabular_table(
                "audio_data", audio_path, "Animal audio data table"
            )
            db.add_tabular_table(
                "image_data", image_path, "Animal image data table"
            )

        else:
            raise ValueError(f"Unsupported use case: {self.use_case}")

        return db

    def get_system_name(self) -> str:
        """Return the name of the system."""
        # SemBench evaluator 用 system_name 给结果文件命名 (e.g. raw_results/caesura/Q1.csv)
        return "caesura"

    def execute_query(self, query_id: int) -> GenericQueryMetric:
        """
        Execute a specific query and return metric object with results.

        Args:
            query_id: ID of the query (e.g., 1 for Q1, 5 for Q5)

        Returns:
            GenericQueryMetric object containing results DataFrame and metrics
        """
        # =====================================================================
        # SemBench 标准 execute_query — 单查询入口
        # =====================================================================
        # 流程:
        #   1. 用反射找 self._execute_q<id> 方法
        #   2. 调用 → 拿到 DataFrame
        #   3. 测执行时间 + 收集 token usage + 算 cost
        #   4. 任何异常 → metric.status = "failed" + 空 DataFrame fallback

        # Create appropriate metric object
        metric = GenericQueryMetric(query_id=query_id, status="pending")

        try:
            # 反射找方法 (e.g. query_id=5 → self._execute_q5)
            query_fn = self._discover_query_impl(query_id)
            start_time = time.time()
            # 实际跑 query (子类实现)
            results = query_fn()
            execution_time = time.time() - start_time

            # Store results in metric
            metric.execution_time = execution_time
            # 防御性 cast: 子类返回非 DataFrame 时 fallback 到空 DataFrame
            metric.results = (
                results if isinstance(results, pd.DataFrame) else pd.DataFrame()
            )
            metric.status = "success"

            # Get token usage and cost from execution
            # 从 L1 patched MyOpenAI 拿 token counter (见 model.py L1 MODIFY)
            self._update_token_usage_and_cost(metric, results)

        except Exception as e:
            # Handle failure
            # 失败时不抛出, 而是返回 failed metric — 让 SemBench 继续跑其它 query
            metric.status = "failed"
            metric.error = str(e)
            metric.results = self._get_empty_results_dataframe(query_id)
            print(f"  Error in Q{query_id} execution: {type(e).__name__}: {e}")
            # 打印完整 traceback 给 debug; SemBench evaluator 不读 stdout, 只读 metric
            traceback.print_exc()

        return metric

    def _discover_query_impl(self, query_id: int):
        """
        Discover and return the implementation for a specific query.

        Args:
            query_id: ID of the query to find implementation for

        Returns:
            Callable query implementation method

        Raises:
            NotImplementedError: If query implementation is not found
        """
        # 反射: 看 self 是否有 _execute_q<id> 方法; 有则返回该方法
        # 这种 dynamic dispatch 让子类 (movie/caesura_runner.py 等) 只需实现 _execute_q1..N 方法,
        # 而不必维护 query_id → method 的显式 dict
        method_name = f"_execute_q{query_id}"
        if hasattr(self, method_name):
            return getattr(self, method_name)
        else:
            raise NotImplementedError(
                f"Query Q{query_id} implementation not found. "
                f"Please implement {method_name} method."
            )

    def _discover_queries(self) -> List[int]:
        """
        Discover available queries for CAESURA.

        Any method named ``_execute_q<i>`` (where <i> is an integer ≥1) is treated
        as an implemented query.  The function returns the list of those integer
        IDs in ascending order.

        Returns:
            List of available query IDs
        """
        # 反射变体: 扫描所有 _execute_q<N> 方法名, 返回 sorted [N1, N2, ...] 列表
        # SemBench runner 用这个来判断 "本 system 支持哪些 query" — 后续可以 batch 跑全部
        pattern = re.compile(r"_execute_q(\d+)$")
        query_ids: List[int] = []

        # `dir(self)` lists all attribute names visible on the instance;
        # we then pick out callables whose names match our pattern.
        # dir() 列出 instance 上所有属性名 (含父类继承); pattern.match 抓 _execute_q<digits>
        for attr_name in dir(self):
            match = pattern.match(attr_name)
            if match:
                attr = getattr(self, attr_name, None)
                # callable() 检查是否是函数/方法 — 防止误把 _execute_q1 = "foo" 这种字段也加入
                if callable(attr):
                    query_ids.append(int(match.group(1)))

        return sorted(query_ids)

    def _get_empty_results_dataframe(self, query_id: int) -> pd.DataFrame:
        """
        Get empty DataFrame with correct columns for a query.
        Should be overridden by scenario-specific runners.

        Args:
            query_id: ID of the query

        Returns:
            Empty DataFrame with appropriate columns
        """
        # 默认空 DataFrame — 子类 (e.g. movie/caesura_runner.py) 覆盖此方法返回 *有列名* 的空 df
        # SemBench evaluator 在 failed query 上仍然需要 *正确 schema* 的空 df 用于结果对齐
        # Generic empty DataFrame - should be overridden by subclasses
        return pd.DataFrame()

    def _update_token_usage_and_cost(
        self, metric: GenericQueryMetric, results: pd.DataFrame
    ):
        """Update metric with token usage and cost information."""
        # =====================================================================
        # 从 L1 patched MyOpenAI 拿 token usage → 算实际 cost
        # =====================================================================
        # 依赖 L1 patches in caesura/main.py + caesura/model.py:
        #   Caesura.get_token_usage()   → dict(prompt_tokens, completion_tokens, total_tokens)
        try:
            # Get actual token usage from the CAESURA agent
            token_usage_dict = self.caesura_agent.get_token_usage()
            metric.token_usage = token_usage_dict.get("total_tokens", 0)
            metric.money_cost = self._calculate_actual_cost(token_usage_dict)

            # Print usage for debugging
            # 打印实时 usage 供 dev 监控 (long-running batch 时方便看进度)
            print(
                f"  Token usage: {metric.token_usage} tokens (prompt: {token_usage_dict.get('prompt_tokens', 0)}, "
                f"completion: {token_usage_dict.get('completion_tokens', 0)}), Cost: ${metric.money_cost:.4f}"
            )

        except Exception as e:
            # 失败兜底: 主要可能 trigger 是 langchain SQLiteCache 命中 — patched _generate 跳过 token 累加
            # 详 [LOG.md pitfall #6](../../../../AllSQPE/CAESURA/LOG.md)
            print(f"  Warning: Could not get token usage: {e}")
            metric.token_usage = 0
            metric.money_cost = 0.0

    def _calculate_actual_cost(self, token_usage_dict: dict) -> float:
        """Calculate actual cost based on real token usage from API."""
        # =====================================================================
        # 按 OpenAI 0613 模型定价算 cost
        # =====================================================================
        # ⚠ 价格 hardcode 至 2023-06 时期; OpenAI 后续多次降价
        # ⚠ 不支持其它 model (Gemini / Claude); _map_model_name 把所有 input 都映射成 0613, 所以这里总是落到这 2 个之一
        # Token pricing (USD per 1K tokens)
        pricing = {
            "gpt-3.5-turbo-0613": {"input": 0.0015, "output": 0.002},
            "gpt-4-0613": {"input": 0.03, "output": 0.06},
        }

        if self.caesura_model in pricing:
            prompt_tokens = token_usage_dict.get("prompt_tokens", 0)
            completion_tokens = token_usage_dict.get("completion_tokens", 0)

            model_pricing = pricing[self.caesura_model]
            # 定价公式: input * (prompt / 1000) + output * (completion / 1000)
            cost = (prompt_tokens / 1000) * model_pricing["input"] + (
                completion_tokens / 1000
            ) * model_pricing["output"]
            # round 到 6 位小数 (~ micro-cent 精度)
            return round(cost, 6)

        return 0.0

    def execute_caesura_query(self, query_text: str) -> pd.DataFrame:
        """
        Execute a CAESURA natural language query and return results DataFrame.
        This method can be used by specific query implementations.

        Args:
            query_text: Natural language query string

        Returns:
            Results DataFrame
        """
        # =====================================================================
        # 子类 _execute_q<N> 方法的核心 helper — 跑一次 NL query 拿 DataFrame
        # =====================================================================
        # 流程 (依赖 L1 patches in caesura/main.py):
        #   1. reset_for_new_query() 清 chat history / working set / token counter
        #   2. run(query_text) 跑 4 阶段流水线
        #   3. get_final_result() 取 Caesura.last_result (= database.final_result())
        #   4. .data_frame.copy() 拿 DataFrame (copy 避免后续 mutate 影响 working set)
        try:
            # Reset agent state for new query
            # L1 抽象核心 — 让 agent 跨 query 复用而不污染状态
            self.caesura_agent.reset_for_new_query()

            # Execute query using the shared agent
            self.caesura_agent.run(query_text)
            final_result = self.caesura_agent.get_final_result()

            # Extract results
            # final_result 是 Table 实例 (或 None — 当 plan 一步都没跑成功时)
            # hasattr 防御 — 万一 final_result 不是 Table 而是 None / DataFrame, fallback 空 df
            if final_result is not None and hasattr(final_result, "data_frame"):
                result_df = final_result.data_frame.copy()
                return result_df
            else:
                return pd.DataFrame()

        except Exception as e:
            # 异常吞掉返回空 df; 不抛 — 让 execute_query 的 outer try/except 捕获 metric 状态
            # ⚠ 但这里已经吞了一层! 外层 execute_query 看不到 traceback, 只看到 empty df
            # 这意味着 metric.status 仍是 "success" + 空 results — 容易误判
            print(f"Error executing CAESURA query: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def get_available_use_cases(self) -> List[str]:
        """Get list of available use cases."""
        # ⚠ "detective" 和 "animals" 在这里报出, 但实际没有对应的 per-scenario subclass
        # 详 LOG.md "遗留问题 #3" — dead reference, 应只列 ["movie"]
        return ["movie", "detective", "animals"]

    def describe_dataset(self) -> str:
        """Get description of the current dataset."""
        # 转发到 Database.describe() — 返回 LLM 看的 schema 描述字符串
        return self.database.describe()
