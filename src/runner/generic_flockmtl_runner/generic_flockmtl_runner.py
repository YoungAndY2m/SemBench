"""
DuckDB FlockMTL runner implementation.

============================================================================
SemBench L1 wrapper — generic_flockmtl_runner.py
============================================================================
教学注释 pass (L1 ADD) by Claude.
对应 [FlockMTL LOG_STRUCTURE.md §10.1.1](../../../../AllSQPE/FlockMTL/LOG_STRUCTURE.md#1011-l1-add--新增文件l0-没有的).

----------------------------------------------------------------------------
这个文件干什么 (一句话)
----------------------------------------------------------------------------
**最小适配器** — 把 FlockMTL 的 "在 DuckDB 内跑 SQL with LLM scalar" 包装成 SemBench
`GenericRunner` 接口. 与 CAESURA L1 wrapper 复杂度对比:
  - CAESURA L1: vendored 整包 + 3 文件 patch + 4 add + schema-adapter (1191 LoC)
  - FlockMTL L1: 0 vendored + 0 patch + 仅 8 个 Python 文件 (1039 LoC), 全是模板化 boilerplate

----------------------------------------------------------------------------
L0 → L1 抽象增益 (3 件事)
----------------------------------------------------------------------------
1. **Per-scenario Setup 类**: 把 "INSTALL flockmtl + LOAD + CREATE SECRET + CREATE MODEL + load CSV"
   一套 boilerplate 封装成 4 个近重复 Python 类 (movie / medical / mmqa / cars).
2. **`<<model_name>>` Jinja2 SQL 模板**: 用 `--model` CLI 参数注入 SQL 文件.
3. **GenericRunner 接口适配**: 让 FlockMTL 的 DuckDB conn execute 对齐 SemBench 标准 API.

----------------------------------------------------------------------------
关键设计: Jinja2 `<<>>` 自定义占位符
----------------------------------------------------------------------------
Jinja2 默认 `{{var}}` 占位符与 SQL 内 `{ }` 字符 (e.g. JSON struct) 冲突, 所以改用 `<<var>>`:
  jinja_env = Environment(variable_start_string="<<", variable_end_string=">>")
SQL 模板 (e.g. files/movie/query/flockmtl/Q1.sql):
  SELECT llm_filter({'model_name': '<<model_name>>'}, ...) FROM reviews;
渲染后:
  SELECT llm_filter({'model_name': 'gpt-4o'}, ...) FROM reviews;

----------------------------------------------------------------------------
关键缺失 (LOG.md "遗留问题 #5")
----------------------------------------------------------------------------
**Token usage / cost 硬编码为 0** (line 92-95).
本来可以调 L0 的 `flock_get_metrics()` SQL 函数拿 token + duration + api_calls,
但本 wrapper 没接入 — 是 L1 最大的遗漏 (LOG_STRUCTURE.md §10.3 已记录).

----------------------------------------------------------------------------
零基础读者预备知识 (Python)
----------------------------------------------------------------------------
- `from overrides import override`:
    第三方库, 装饰器 `@override` 强制子类方法 *必须 override 父类*; 类似 Java `@Override`.
    Python 内置没这个 (PEP 698 已加但仍要 import); 用 `overrides` 库填空.

- `from jinja2 import Environment`:
    Jinja2 = Python 模板引擎; Environment 类是模板渲染的配置容器.
    variable_start_string / variable_end_string: 改默认 `{{}}` 占位符的开闭符号.

- `jinja_env.from_string(template).render(var=value)`:
    一次性的模板渲染 — 用 SQL 字符串当模板, 把 `<<model_name>>` 替换成实际值.

- `self.flockmtl_conn = None` (init 时未连):
    实际 conn 由子类 (e.g. FlockMTLRunner in movie/runner/flockmtl_runner.py) 的 __init__ 内
    调 SetupFlockMTLXxx().get_connection() 赋值.

- `GenericRunner` (从 runner.generic_runner import):
    SemBench 全部 system runner 的统一基类; 提供 use_case / model_name / scale_factor /
    _discover_query_text / scenario_handler 等共享字段 + execute_queries 抽象方法.
============================================================================
"""

import time
from typing import Dict, List

from jinja2 import Environment
from overrides import override

from runner.generic_runner import GenericQueryMetric, GenericRunner

# Jinja2 自定义 `<<>>` 占位符 — 避免与 SQL 内 `{ }` 冲突 (FlockMTL 的 STRUCT 字面量用 `{...}`)
jinja_env = Environment(variable_start_string="<<", variable_end_string=">>")


class GenericFlockMTLRunner(GenericRunner):
    """Runner for FlockMTL."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gpt-4o",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        """
        Initialize DuckDB FlockMTL runner.

        Args:
            use_case: The use case to run
            model_name: LLM model to use
        """
        # 注意! concurrent_llm_worker=20 默认值 *不被 FlockMTL 用* —
        # FlockMTL 内部 (base_handler.cpp curl_multi) 自己管并发, 这参数纯 boilerplate
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )
        # flockmtl_conn 留空; 子类 (per-scenario Runner) 必须在自己的 __init__ 内填 DuckDB conn
        self.flockmtl_conn = None

    @override
    def get_system_name(self) -> str:
        return "flockmtl"

    @override
    def execute_queries(
        self, query_ids: List[int]
    ) -> Dict[int, GenericQueryMetric]:
        # =====================================================================
        # 主入口 — 批量跑 N 个 query, 返回 Dict[query_id, GenericQueryMetric]
        # =====================================================================
        # 流程: 抽 SQL 文本 → Jinja2 渲染 <<model_name>> → DuckDB conn.execute → 收 DataFrame
        # 第 1 步: 抽 query 文本 (从 SQL 文件 或 scenario_handler — 后者支持更复杂场景如 mmqa)
        # `scenario_handler` 是 GenericRunner 内 optional 字段, 若不为 None 则走它取 query (overrideable);
        # 默认 None → 走 `_discover_query_text` (基于约定路径 files/<use_case>/query/<system>/Q<id>.sql)
        query_texts = {
            query_id: (
                self._discover_query_text(query_id)
                if self.scenario_handler is None
                else self.scenario_handler.get_query_text(
                    query_id, self.get_system_name()
                )
            )
            for query_id in query_ids
        }
        query_metrics = {
            query_id: GenericQueryMetric(query_id=query_id, status="pending")
            for query_id in query_ids
        }

        for query_id, query_text in query_texts.items():
            try:
                # Replace variable names in the query text
                # Jinja2 渲染 <<model_name>> → self.model_name 实际值
                # 例: `llm_complete({'model_name': '<<model_name>>'}, ...)` → `llm_complete({'model_name': 'gpt-4o'}, ...)`
                templated_query = jinja_env.from_string(query_text).render(
                    model_name=self.model_name
                )

                # 调试输出 — print 渲染后的 SQL 让 dev 看
                print(templated_query)

                # 真正跑 SQL — DuckDB conn.execute 异步触发 LLM 调用 (FlockMTL L0 内部并发)
                start_time = time.time()
                query_job = self.flockmtl_conn.execute(templated_query)
                # fetchdf: DuckDB conn API, 把结果转 pandas DataFrame
                df = query_job.fetchdf()
                execution_time = time.time() - start_time

                # 成功路径
                query_metrics[query_id].results = df
                query_metrics[query_id].execution_time = execution_time
                query_metrics[query_id].status = "success"
            except Exception as e:
                # 任何异常 (SQL 语法 / LLM API 错 / DuckDB 内部) → failed metric, 继续下一 query
                print(
                    f"  Error executing query {query_id}: {type(e).__name__}: {e}"  # noqa: E501
                )
                query_metrics[query_id].status = "failed"
                query_metrics[query_id].error = str(e)
        # 调试 print 整 dict (没生产价值; dev 阶段留)
        print(query_metrics.items())
        # ⚠ 关键缺失: token usage / cost 都硬编码为 0
        # 实际可以调 SELECT flock_get_metrics() 拿 token + duration + api_calls — 但本 wrapper 没实现
        # 详 [FlockMTL/LOG.md "遗留问题 #5"](../../../../AllSQPE/FlockMTL/LOG.md)
        for query_id, metrics in query_metrics.items():
            if metrics.status != "failed":
                # TODO: Implement token usage and cost calculation
                # Currently no way to extract token usage from FlockMTL
                metrics.token_usage = 0
                metrics.money_cost = 0.0

        return query_metrics
