"""
Created on May 28, 2025

@author: Jiale Lao

Generic runner base class for all data system runners, you should implement
execute_query(query_id: int) -> GenericQueryMetric in your system
implementations

============================================================================
教学注释 pass (CLAUDE.md §5.5 全面注释)
============================================================================

本文件定义 SemBench *runner 层* 的根抽象, 是全 SQPE (lotus / palimpzest /
bigquery / thalamusdb / flockmtl / caesura / snowflake) runner 的共同父类.

两个主体类
----------
  GenericQueryMetric (@dataclass)  单 query 的执行结果容器
  GenericRunner      (ABC)         所有 system runner 的根抽象类

设计模式
--------
1. **Template Method 模板方法**: run_all_queries 是骨架, 调
   execute_queries 调 execute_query — 子类可以:
     (a) 只 override execute_query (单 query) — 默认 execute_queries
         循环调它.
     (b) override execute_queries (批量) — 适合 LOTUS / Palimpzest 这种
         自家支持 batch 优化的 system.
2. **Abstract Base Class (ABC, Python 抽象基类)**: get_system_name 必须
   override (@abstractmethod 强制), 实例化未 override 的子类会立刻
   TypeError.
3. **@dataclass**: Python 3.7+ 引入. 在类上加 @dataclass 装饰器自动 gen
   __init__ / __repr__ / __eq__, 字段声明用类属性 + type hint, 默认值
   用 field(default_factory=...) (避免可变默认参数陷阱).

路径布局 (self.<x>_path 字段)
-----------------------------
  self.base_path        = <repo_root>         # __file__ 上溯 2 级
  self.files_path       = <repo>/files/{use_case}/
  self.data_path        = <repo>/files/{use_case}/data/sf_{scale_factor}/
  self.query_path       = <repo>/files/{use_case}/query/
  self.results_path     = <repo>/files/{use_case}/raw_results/{system}/
  self.metrics_path     = <repo>/files/{use_case}/metrics/

结果落盘 schema
---------------
  raw_results/{system}/Q{id}.csv     每个 query 的 system output DataFrame
  metrics/{system}.json              {Q1: {...}, Q2: {...}}; 含 row_count /
                                     execution_time / token_usage / cost
                                     + 后续 evaluator 写入的 P/R/F1 字段

调用流 (跟 src/run.py 配套)
---------------------------
  run.py  → 实例化 runner_class(use_case=..., ...)
              → __init__ 里调 scenario_handler.setup_scenario(systems)
                  # 让数据 ready (调 scenario.py 的 setup_scenario)
          → runner.run_all_queries(queries=...)
              → execute_queries
                  → execute_query (子类实现, 跑 LLM / DB query)
              → save_metrics  # 写 metrics/{system}.json
          → evaluator.evaluate_system(system, queries=...)
              # 读 metrics/{system}.json + raw_results/{system}/Q{id}.csv
              # 算 P/R/F1 + dict.update 写回同一 JSON

引用: [LOG_STRUCTURE.md §4.1 GenericRunner](../../LOG_STRUCTURE.md)
============================================================================
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================================
# GenericQueryMetric — 单 query 的执行结果容器 (@dataclass)
# ============================================================================
# 为什么用 dataclass:
#   - 写 7 个字段就拿到 __init__ / __repr__ / __eq__ 三个 dunder 方法的
#     auto-gen, 比手写 __init__ 简洁 4x.
#   - asdict(metric) 直接序列化成 dict (后续转 JSON).
#
# 字段语义:
#   query_id          int           Q{id} 的数字 id (e.g. 1, 5, 10)
#   status            'success'/'failed'
#   execution_time    float (sec)   runner 处理该 query 的 wall-clock
#   results           pd.DataFrame  query output (列名/行数因 query 而异)
#   token_usage       int           调用 LLM 累计 token 数 (sum of input +
#                                   output across all sem_filter / sem_map /
#                                   sem_join calls within this query)
#   money_cost        float USD     按 model 单价折算的 dollar cost
#   error             Optional[str] 失败时的 exception message
#
# field(default_factory=pd.DataFrame):
#   不能写 results: pd.DataFrame = pd.DataFrame() — Python 会在 *class
#   definition* 时调一次 pd.DataFrame(), 然后所有实例共享同一个 DataFrame
#   (经典 mutable default argument 陷阱). default_factory 让每次 __init__
#   时 new 一个空 DataFrame, 每个实例独立.
@dataclass
class GenericQueryMetric:
    """Base class for query metrics with results."""

    query_id: int
    status: str  # 'success' or 'failed'
    execution_time: float = None
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    token_usage: int = None
    money_cost: float = None
    error: Optional[str] = None

    # ========================================================================
    # to_dict — 转 JSON-serializable dict (DataFrame 替换为 row_count int)
    # ========================================================================
    # 为什么 strip results DataFrame:
    #   JSON 不能直接 dump pandas DataFrame; 而且把整个 DataFrame 嵌到
    #   metrics.json 会让文件爆炸 (一个 query 可能返回 50K rows).
    #   raw_results/{sys}/Q{id}.csv 已经单独落盘 DataFrame, 这里只留行数
    #   作 quick reference. row_count = len(df) 是 evaluator 算 P/R 时
    #   需要的 base 数字.
    # 过滤 v is not None 让 JSON 干净 (e.g. 失败 query 没 token_usage,
    # 不要在 JSON 里出现 "token_usage": null).
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metric to dictionary (excluding DataFrame; instead includes
        row_count).
        """
        data = {
            k: v
            for k, v in asdict(self).items()
            if v is not None and k != "results"
        }
        data["row_count"] = len(self.results) if self.results is not None else 0
        return data


# ============================================================================
# GenericRunner — 全 SQPE runner 的抽象基类 (ABC)
# ============================================================================
# 子类 (e.g. LotusRunner / PalimpzestRunner / ...) 必须实现:
#   get_system_name() -> str          系统名 (e.g. "lotus")
#   execute_query(qid) -> Metric      跑单 query (除非 override execute_queries)
#
# 可选 override:
#   execute_queries(qids) -> {qid: metric}    批量优化用
#   _get_empty_results_dataframe(qid)         查询失败时返回的默认 schema
#   _discover_queries()                       自动枚举 query (默认从 scenario
#                                             handler 拿)
class GenericRunner(ABC):
    """Base class for all system runners."""

    # ========================================================================
    # __init__ — 设置 path 字段 + 实例化 scenario_handler + 调 setup_scenario
    # ========================================================================
    # 步骤:
    #   1. 设 system_name (调 子类 abstractmethod get_system_name).
    #   2. 设 5 个 path 字段 (base/files/data/query/results/metrics_path).
    #      base_path = __file__.resolve().parents[2]
    #        # __file__ = .../SemBench/src/runner/generic_runner.py
    #        # parents[0] = .../src/runner/
    #        # parents[1] = .../src/
    #        # parents[2] = .../SemBench/ (repo 根)
    #   3. mkdir -p results_path + metrics_path (parents=True 类似 mkdir -p,
    #      exist_ok=True 让重复跑不抛异常).
    #   4. 初始化 self.metrics = {}  (query_id → GenericQueryMetric 字典).
    #   5. 调 get_scenario_handler 拿 scenario 对象, 若 skip_setup=False 则
    #      自动调 setup_scenario([system_name]) 准备数据 (idempotent).
    #
    # concurrent_llm_worker:
    #   控制各 system 内 LLM 并发数 (e.g. LOTUS 跑 batch sem_filter 时同时
    #   开 16 个 OpenAI request). 由各 system runner 自行解释; 父类只把它
    #   存下来 + 写入 metrics JSON 让后续分析能 trace 不同 concurrent setting
    #   下的 throughput.
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str,
        concurrent_llm_worker: int,
        skip_setup: bool = False,
    ):
        """
        Initialize the runner.

        Args:
            use_case: The use case to run (e.g., 'movie')
        """
        self.use_case = use_case
        self.system_name = self.get_system_name()

        # Set up paths
        self.base_path = Path(__file__).resolve().parents[2]
        self.files_path = self.base_path / "files" / use_case
        self.data_path = self.files_path / "data" / f"sf_{scale_factor}"
        self.query_path = self.files_path / "query"
        self.results_path = self.files_path / "raw_results" / self.system_name
        self.metrics_path = self.files_path / "metrics"

        # Create directories if they don't exist
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.metrics_path.mkdir(parents=True, exist_ok=True)

        # Initialize metrics storage
        self.metrics: Dict[int, GenericQueryMetric] = {}
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.concurrent_llm_worker = concurrent_llm_worker

        # Manage scenario-specific data
        self.scenario_handler = GenericRunner.get_scenario_handler(
            self.use_case, self.scale_factor
        )
        if not skip_setup and self.scenario_handler is not None:
            self.scenario_handler.setup_scenario([self.get_system_name()])

    # ========================================================================
    # get_system_name — 抽象方法; 子类必须实现, 返回字面 system 名
    # ========================================================================
    # @abstractmethod 是 ABC 的 hook: 没 override 这个方法的子类被实例化
    # 时立刻 TypeError("Can't instantiate abstract class ..."). 主要作用是
    # *fail loud* — 防止有人忘了实现某个必要 method 后 runtime 才报错.
    # 命名约定: 全小写 + 下划线, 跟 run.py / config / .venvs 目录名一致.
    @abstractmethod
    def get_system_name(self) -> str:
        """Return the name of the system (e.g., 'lotus', 'bigquery')."""
        raise NotImplementedError("Subclasses must implement get_system_name()")

    # ========================================================================
    # execute_query — 默认 stub, 子类 override 之一; 跑单 query 返 Metric
    # ========================================================================
    # 与 get_system_name 不同, 这个不是 @abstractmethod — 因为子类有两个
    # 选择 (override 它, 或 override execute_queries). 默认实现抛
    # NotImplementedError, 提示"实现这个或 override 上层"; ABC + runtime
    # 检查的混合, 给子类灵活度.
    def execute_query(self, query_id: int) -> GenericQueryMetric:
        """
        Execute a specific query and return metric object with results.

        Args:
            query_id: ID of the query (e.g., 1 for Q1, 5 for Q5)

        Returns:
            QueryMetric object containing results DataFrame and metrics
        """
        raise NotImplementedError(
            "Subclasses must either implement execute_query() or override execute_queries()"  # noqa: E501
        )

    # ========================================================================
    # execute_queries — 默认逐个调 execute_query; 子类可 override 做 batch
    # ========================================================================
    # 模板方法的中间层. 默认 for-loop 在主线程串行调; 每个 query 用
    # try/except 包住, 一个 query 挂了不影响后面 query (e.g. Q3 LLM
    # timeout 不应拖累 Q4-Q10).
    #
    # 子类何时 override:
    #   - LOTUS 跑同一个 dataset 的多个 sem_filter, 可以一次 LLM batch
    #     call 处理多个 query 节省 token (override 后跑得快).
    #   - Palimpzest 用 demonstrate 训练 cascade, 可以跨 query 共享样本.
    #   - BigQuery / DuckDB / Snowflake 不 batch (每个 query 独立 SQL),
    #     用默认实现就行.
    def execute_queries(
        self, query_ids: List[int]
    ) -> Dict[int, GenericQueryMetric]:
        """
        Execute multiple queries and return metrics.
        Systems can choose to either override this method if it makes more
        sense to do batch execution, or use the default implementation and
        instead override execute_query(query_id: int).

        Args:
            query_ids: List of query IDs to execute

        Returns:
            Dictionary mapping query IDs to GenericQueryMetric objects
        """
        results = {}
        for query_id in query_ids:
            try:
                results[query_id] = self.execute_query(query_id)
            except Exception as e:
                print(f"Error executing query {query_id}: {e}")
                results[query_id] = GenericQueryMetric(
                    query_id=query_id,
                    execution_time=0.0,
                    status="failed",
                    error=str(e),
                )
        return results

    # ========================================================================
    # run_all_queries — 模板方法顶层 (run.py 主入口调它)
    # ========================================================================
    # 流程:
    #   1. 若 queries=None → 调 _discover_queries() 拿全集.
    #   2. 调 execute_queries(queries) → 跑跑跑, 拿 {qid: metric}.
    #   3. self.metrics = result; 调 save_metrics() 落盘.
    # 返回值: self.metrics dict.
    #
    # 注意: save_metrics 在这里调 → 跑完 *所有* query 才一次性 dump JSON.
    # 中途 crash 会丢失 metric. 工程上可以改成每 query 跑完就 incremental
    # save, 但要权衡 I/O 成本. 当前实现接受 "all-or-nothing" 失败模式.
    def run_all_queries(
        self, queries: Optional[List[int]] = None
    ) -> Dict[int, GenericQueryMetric]:
        """
        Run all queries for this system.

        Args:
            queries: Optional list of specific query IDs to run

        Returns:
            Dictionary mapping query IDs to metrics
        """
        # If no specific queries provided, discover all available queries
        if queries is None:
            queries = self._discover_queries()

        print(f"\nRunning {len(queries)} queries for {self.system_name}")
        self.metrics = self.execute_queries(queries)
        self.save_metrics()

        return self.metrics

    # ========================================================================
    # get_query_text — 父类 helper, 从 query/<query_type>/Q{id}.txt 读文本
    # ========================================================================
    # 注意: 这个 helper 用 *父类* path 约定 (query/{type}/Q{id}.txt),
    # 与 scenario.py 里各 ScenarioHandler.get_query_text 的路径
    # (query/{system}/Q{id}.*) *不同*. 通常 runner 直接走 scenario_handler.
    # 这个父类版本是早期 (movie / detective) scenario 用的, 新 scenario
    # 走 scenario.get_query_text. 保留兼容性.
    def get_query_text(
        self, query_id: int, query_type: str = "natural_language"
    ) -> str:
        """
        Read query text from file.

        Args:
            query_id: ID of the query (e.g., 1 for Q1, 5 for Q5)
            query_type: Type of query ('mm_sql', 'natural_language', etc.)

        Returns:
            Query text as string
        """
        query_name = f"Q{query_id}"
        query_file = self.query_path / query_type / f"{query_name}.txt"
        if not query_file.exists():
            raise FileNotFoundError(f"Query file not found: {query_file}")

        with open(query_file, "r") as f:
            return f.read().strip()

    # ========================================================================
    # save_results — 把单 query 的 DataFrame 落盘到 raw_results/{sys}/Q{id}.csv
    # ========================================================================
    # 命名约定 (与 ScenarioHandler.get_query_text 大写 Q 一致):
    #   raw_results/lotus/Q1.csv / raw_results/bigquery/Q1.csv / ...
    # CSV 是跨 system 中立格式, evaluator 读 CSV 时不用关心是 LOTUS DataFrame
    # 还是 BigQuery 返回; 同样的 schema 直接比对.
    def save_results(self, query_id: int, results: pd.DataFrame):
        """
        Save query results to CSV file.

        Args:
            query_id: ID of the query
            results: DataFrame containing results
        """
        query_name = f"Q{query_id}"
        output_file = self.results_path / f"{query_name}.csv"
        results.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

    # ========================================================================
    # save_metrics — 写 metrics/{system}.json + 每 query CSV
    # ========================================================================
    # 步骤:
    #   1. 对每个 (qid, metric):
    #        a. metric.to_dict() 拿 JSON-safe dict.
    #        b. 注入 model_name + concurrent_llm_worker (这两个在 metric
    #           本身没有, 由 runner config 决定, 写在 JSON 里方便后续
    #           cross-config 对比).
    #        c. save_results(qid, metric.results) 把 DataFrame 单独写 CSV.
    #   2. json.dump 整个 dict 到 metrics/{system}.json (indent=2 让人读).
    #
    # ★ 关键设计: 写整个 dict 一次, *不是* update. 同一 system 重跑会
    # 完全覆盖前次结果. 后续 evaluator.evaluate_system 用 dict.update()
    # 模式追加 P/R/F1 字段 (它先 json.load 现有 JSON, update, 再写回),
    # 不会冲掉这里写的 row_count / time / token / cost.
    def save_metrics(self):
        """Save metrics to JSON file."""
        metrics_file = self.metrics_path / f"{self.system_name}.json"

        # Convert metrics to dict format
        metrics_dict = {}
        for query_id, metric in self.metrics.items():
            query_name = f"Q{query_id}"
            metrics_dict[query_name] = metric.to_dict()
            metrics_dict[query_name]["model_name"] = self.model_name
            metrics_dict[query_name][
                "concurrent_llm_worker"
            ] = self.concurrent_llm_worker
            self.save_results(query_id, metric.results)

        # # write query results to csv files
        # for query_id, metric in self.metrics.items():
        #     self.save_results(query_id, metric.results)

        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")

    # ========================================================================
    # _get_empty_results_dataframe — query 失败时返回的 placeholder DataFrame
    # ========================================================================
    # 默认返空 DataFrame; 子类可 override 返回带正确列名 / dtype 的空 df
    # (让 evaluator 的 schema 检查通过). 例如 movie scenario Q1 期望列名
    # ['movie_id', 'title'], 失败时返回 pd.DataFrame(columns=['movie_id',
    # 'title']) 让 evaluator 算 0/N precision 而不是 schema mismatch error.
    def _get_empty_results_dataframe(self, query_id: int) -> pd.DataFrame:
        """
        Get empty DataFrame with correct columns for a query.
        Override in subclasses for query-specific schemas.

        Args:
            query_id: ID of the query

        Returns:
            Empty DataFrame with correct columns
        """
        return pd.DataFrame()

    # ========================================================================
    # _discover_queries — 没传 --queries 时自动列出全部 query ID
    # ========================================================================
    # 优先级:
    #   1. scenario_handler.discover_available_queries(system_name) — 让
    #      scenario 决定 (mmqa / animals / movie / ecomm 都实现了这个方法).
    #   2. 没 scenario handler 或 handler 没该方法 → 退回到查
    #      query/mm_sql/Q*.txt 目录 (老 scenario detective 用过这条路径).
    #   3. 都没 → 返 [] (调用方应该提示用户 --queries 是必需的).
    def _discover_queries(self) -> List[int]:
        """
        Discover available queries for this system.
        Default implementation looks for query files.

        Returns:
            List of query IDs
        """
        # If we can, go via the scenario handler because it knowns best how
        # queries are structured for the scenario
        scenario_handler = GenericRunner.get_scenario_handler(
            self.use_case, self.scale_factor
        )
        if scenario_handler is not None:
            return scenario_handler.discover_available_queries(
                system_name=self.get_system_name()
            )

        # Look in mm_sql directory by default
        mm_sql_path = self.query_path / "mm_sql"
        if mm_sql_path.exists():
            query_files = list(mm_sql_path.glob("Q*.txt"))
            query_ids = []
            for f in query_files:
                try:
                    # Extract number from filename like Q1.txt
                    query_id = int(f.stem[1:])
                    query_ids.append(query_id)
                except ValueError:
                    continue
            return sorted(query_ids)

        return []

    # ========================================================================
    # _discover_query_text — 尝试读 query 文件; 大小写 q 都试一遍
    # ========================================================================
    # 工程上 case-insensitive fallback: 先试 Q1.sql, 不在再试 q1.sql.
    # 防御 mmqa scenario 大小写不一致问题 (见 mmqa_scenario.py 顶部注释).
    def _discover_query_text(self, query_id: int) -> str:
        expected_path = os.path.join(
            self.query_path, self.system_name, f"Q{query_id}.sql"
        )
        if not os.path.exists(expected_path):
            expected_path = os.path.join(
                self.query_path, self.system_name, f"q{query_id}.sql"
            )
            if not os.path.exists(expected_path):
                raise FileNotFoundError(
                    f"Query file not found for {self.system_name} q{query_id}"
                )

        with open(expected_path, "r") as f:
            return f.read().strip()

    # ========================================================================
    # _discover_query_impl — 找子类的 _execute_q{N} 方法; 用 getattr 反射
    # ========================================================================
    # 这是一种 *per-query* 方法路由: 子类如果不想用 dispatcher pattern,
    # 可以直接定义 def _execute_q1(self): ..., def _execute_q2(self): ...
    # 父类用 getattr(self, "_execute_q{qid}") 反射拿到 callable.
    # 找不到 → NotImplementedError (而非默认 AttributeError, 错误信息更友好).
    def _discover_query_impl(self, query_id) -> callable:
        method_name = f"_execute_q{query_id}"
        try:
            query_fn = getattr(self, method_name)
            if not callable(query_fn):
                raise TypeError(f"{method_name} exists but is not callable")
            return query_fn
        except AttributeError:
            raise NotImplementedError(
                f"Query {query_id} not implemented for {self.system_name}."
            )

    # ========================================================================
    # load_data — 从 self.data_path 读 CSV
    # ========================================================================
    # 简单 wrapper: 拼路径 + pd.read_csv. **kwargs 透传给 read_csv (允许
    # 子类指定 dtype / parse_dates / chunksize 等).
    def load_data(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load data file from the data directory.

        Args:
            filename: Name of the data file

        Returns:
            DataFrame containing the data
        """
        data_file = self.data_path / filename
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        return pd.read_csv(data_file, **kwargs)

    # ========================================================================
    # get_scenario_handler — use_case → ScenarioHandler 的 dispatcher (类似
    #                        run.py:get_runner_class 但反向, runner ↔ scenario)
    # ========================================================================
    # ★ 这个方法 *没* @staticmethod 装饰器但也没用 self — 是 Python 的
    # *unbound method* 风格. 调用方既能 GenericRunner.get_scenario_handler(...)
    # (类方法风格) 也能 runner_instance.get_scenario_handler(...) (实例方法
    # 风格). 这是个 *疏忽性 design*, 严格应加 @staticmethod, 但实际功能 OK,
    # 父类 __init__ 里就用 `GenericRunner.get_scenario_handler(...)` (类调用).
    #
    # 与 run.py:get_runner_class 对称结构:
    #   run.py 是 system → runner_class 的映射
    #   这里是 use_case → scenario_handler_instance 的映射
    # 加新 use_case 时改这里 (类似 §7.2 加新 scenario, [LOG_STRUCTURE.md §7.2]).
    #
    # detective use_case 返 None — 老 scenario, 没 scenario handler, 走父类
    # 的 _discover_queries fallback (扫 query/mm_sql/Q*.txt).
    def get_scenario_handler(use_case: str, scale_factor: int = None):
        """
        Get the scenario handler for a specific use case.
        Dynamically imports the scenario handler based on the use case.
        """
        if use_case == "ecomm":
            from scenario.ecomm.ecomm_scenario import EcommScenario

            return EcommScenario(scale_factor=scale_factor)
        elif use_case == "medical":
            from scenario.medical.medical_scenario import MedicalScenario

            return MedicalScenario(scale_factor=scale_factor)
        elif use_case == "animals":
            from scenario.animals.animals_scenario import AnimalsScenario

            return AnimalsScenario(scale_factor=scale_factor)
        elif use_case == "mmqa":
            from scenario.mmqa.mmqa_scenario import MMQAScenario

            return MMQAScenario(scale_factor=scale_factor)
        elif use_case == "movie":
            from scenario.movie.movie_scenario import MovieScenario

            return MovieScenario(scale_factor=scale_factor)
        elif use_case == "detective":
            return None  # Scenario does not have a specific handler
        elif use_case == "cars":
            from scenario.cars.cars_scenario import CarsScenario

            return CarsScenario(scale_factor=scale_factor)
        else:
            raise ValueError(f"Unknown use case: {use_case}.")
