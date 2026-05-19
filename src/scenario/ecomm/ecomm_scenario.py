"""
============================================================================
教学注释 pass (CLAUDE.md §5.5 全面注释) — ecomm_scenario.py
============================================================================

Ecomm scenario 在 SemBench 里的角色
-----------------------------------
E-commerce (electronic commerce, "电商") 产品评论数据集. 默认 44K rows
左右 (从 Amazon Reviews 派生). 本 scenario 是 SemBench 论文 §6 主实验
图表大量引用的 case.

★ Ecomm 是 6 个 scenario 里 *最 elaborate* 的设计 — 与其它 5 个简单的
{Name}Scenario 不同, ecomm 采用 "下一代" 设计模式:
  1. *TOML 驱动*: query 集合用 TOML 配置 (queries/q{N}.toml) 而不是纯
     文本; 每个 TOML 文件包含:
        [metadata]      query_id, description, accuracy_metric, ...
        [definition]    ground_truth (DuckDB SQL), 各 system dialect 入口
     这让 ground truth + accuracy measure 集中在一处, 不用 hard-code 到
     evaluator.
  2. *Per-system dialects/ 目录*: query 文件按 queries/dialects/{system}/
     组织 (而非其它 scenario 的 query/{system}/).
  3. *get_ground_truth()* 在 scenario 类里直接跑 DuckDB SQL 产生 ground
     truth (而其它 scenario 把 ground truth 推给 evaluator 或在
     generate_data 里预先 dump).
  4. *get_accuracy_measure_for_query()* 让 evaluator 知道每个 query 该
     用哪种 metric (precision / recall / f1 / relative_error / ...).

为什么 ecomm 设计更复杂?
  - SemBench 论文 §5.1 强调 *per-query accuracy metric*: 不同 query 类型
    (selection / aggregation / ranking) 需要不同 metric. ecomm 是第一个
    把这点 *声明式化* 的 scenario.
  - 其它 scenario 还停留在 *hard-code-in-evaluator* 阶段, ecomm 推动 TOML
    式 declarative spec — 推测后续 scenario 会逐步迁移到这个模式.

Scenario Handler 抽象 (通用部分) 见 cars/cars_scenario.py:1-50, 但 ecomm
多出 3 个方法 (_load_queries / get_ground_truth /
get_accuracy_measure_for_query).

依赖说明
--------
- tomli: TOML 文件解析库 (Python 3.11+ 自带 tomllib, 老版本要 tomli pkg).
- duckdb: 内存 SQL 引擎, 用来跑 ground truth SQL. 关键设置:
    set file_search_path = '<data_dir>'
  让 SQL 能用 *相对路径* 引用 CSV (e.g. `SELECT * FROM 'products.csv'`),
  否则 DuckDB 不知道去哪找文件.

引用: [LOG_STRUCTURE.md §5.3 Scenario 层 — ecomm](../../../LOG_STRUCTURE.md)
============================================================================
"""

import os
from typing import Any, Dict, List
import pandas as pd
from .preparation.generate_data import prepare_data
import glob
import tomli
import duckdb


ECOMM_FILES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "files", "ecomm")
)


# ============================================================================
# EcommScenario — ecomm use case 的 ScenarioHandler (7 个方法, 最多)
# ============================================================================
# 与其它 5 个 scenario 不同, ecomm 在 __init__ 里就 *eagerly* 调
# _load_queries 加载所有 TOML query 配置 → self.queries: {query_id: dict}.
# 这让后续 get_ground_truth / get_accuracy_measure_for_query 可以从内存
# 里直接拿, 不用每次重读磁盘.
class EcommScenario:
    """
    E-commerce scenario for benchmarking SQL query performance.

    This class handles scenario-specficic things like:
     * downloading and setting up the dataset
     * discovering available queries
     * retrieving query text for specific systems
     * retrieving ground truth results for queries
    """

    def __init__(self, scale_factor: int = None):
        # Path to the directory where the dataset is stored. This depends on various factors like the scale factor.
        self.data_dir = None
        self.queries = self._load_queries()
        self.scale_factor = scale_factor

    # ========================================================================
    # _load_queries — 读 queries/*.toml, 解析成 {query_id: query_data dict}
    # ========================================================================
    # TOML 文件格式 (示例):
    #   [metadata]
    #   query_id        = 1
    #   description     = "Find products with positive reviews"
    #   accuracy_metric = "f1-score"
    #   [definition]
    #   ground_truth = """SELECT ... FROM 'products.csv' WHERE ..."""
    #
    # 防御:
    #   - TOMLDecodeError → raise ValueError (TOML 语法错误立刻 fail loud).
    #   - 缺 metadata.query_id → 打 warning + skip 这个文件 (不 fail), 因为
    #     有时存在 *草稿 toml*, 不想阻塞其它 query 加载.
    #   - 文件名与 query_id 不匹配 (e.g. q1.toml 但 metadata 写 query_id=2)
    #     → 打 warning + 仍然按 metadata 注册 (metadata 是 source of truth).
    def _load_queries(self) -> Dict[int, Any]:
        """
        Load queries from the e-commerce scenario directory.

        Returns:
            List of query definitions.
        """
        query_files = glob.glob(
            os.path.join(ECOMM_FILES_DIR, "queries", "*.toml")
        )
        queries = {}
        for file in query_files:
            with open(file, "rb") as f:
                try:
                    query_data = tomli.load(f)
                except tomli.TOMLDecodeError as e:
                    raise ValueError(f"Error decoding TOML file {file}: {e}")

                if "query_id" in query_data["metadata"]:
                    queries[query_data["metadata"]["query_id"]] = query_data

                    # The file name should be of the format q1.toml, q2.toml, etc.
                    # Double-check the file name here and print a warning if it doesn't match (no need to enforce strict naming).
                    # This constraint can be relaxed if needed.
                    expected_file_name = (
                        f"q{query_data['metadata']['query_id']}.toml"
                    )
                    if os.path.basename(file) != expected_file_name:
                        print(
                            f"Warning: File {file} does not match expected naming convention {expected_file_name}."
                        )
                else:
                    print(
                        f"Warning: 'metadata.query_id' key not found in {file}. Skipping this query."
                    )
        return queries

    # ========================================================================
    # get_data_dir — 与 animals / mmqa 同, lazy fallback 到 sf_{N}/ 路径
    # ========================================================================
    def get_data_dir(self) -> str:
        if self.data_dir:
            return self.data_dir
        else:
            return os.path.abspath(
                os.path.join(
                    ECOMM_FILES_DIR,
                    "data",
                    f"sf_{str(self.scale_factor)}",
                )
            )

    # ========================================================================
    # setup_scenario — 准备数据 + 灌入 5 个 system (bigquery/snowflake/...)
    # ========================================================================
    # 与其它 scenario 不同: prepare_data 直接返回 data_dir (而非 set 字段),
    # 所以这里 self.data_dir = prepare_data(...) 一行搞定.
    # 不接 flockmtl / caesura / mmqa.
    def setup_scenario(self, systems: List[str]) -> None:
        # Download and prepare data
        self.data_dir = prepare_data(scale_factor=self.scale_factor)

        # Load data into the specified systems
        for system in systems:
            if system == "bigquery":
                from .setup.bigquery import BigQueryEcommSetup

                setup = BigQueryEcommSetup()
                setup.setup_data(self.data_dir)
            elif system == "snowflake":
                from .setup.snowflake import SnowflakeEcommSetup

                setup = SnowflakeEcommSetup()
                setup.setup_data(self.data_dir)
            elif system == "thalamusdb":
                from .setup.thalamusdb import ThalamusDBEcommSetup

                setup = ThalamusDBEcommSetup()
                setup.setup_data(self.data_dir)
            elif system == "lotus":
                pass  # Nothing to do. LOTUS works on raw files.
            elif system == "palimpzest":
                pass  # Nothing to do. Palimpzest works on raw files.
            else:
                raise ValueError(f"Unsupported system: {system}")

    # ========================================================================
    # discover_available_queries — 与其它 scenario 同, 但走 dialects/ 子目录
    # ========================================================================
    # system_name=None  → 返回 self.queries 的全部 key (即所有 TOML 注册的
    #                     query, 不管该 system 是否实现).
    # system_name='X'   → 进一步过滤: 只保留 queries/dialects/X/q{id}.* 存在
    #                     的 query (即该 system 实际有实现的). 这让 evaluator
    #                     知道 ground truth 集合 (全部) 与 system run 集合
    #                     (system 实现) 的差.
    def discover_available_queries(self, system_name: str = None) -> List[int]:
        """
        Discover available queries for the e-commerce scenario.

        Returns:
            List of query IDs.
        """
        all_queries = self.queries.keys()
        if system_name is None:
            return all_queries

        system_query_dir = os.path.abspath(
            os.path.join(ECOMM_FILES_DIR, "queries", "dialects", system_name)
        )
        system_queries = []
        for query_id in all_queries:
            matching_files = glob.glob(
                os.path.join(system_query_dir, f"q{query_id}.*")
            )
            if matching_files:
                system_queries.append(query_id)
        return system_queries

    # ========================================================================
    # get_query_text — 走 queries/dialects/{system}/q{id}.* (而非 query/{sys}/)
    # ========================================================================
    # 与其它 scenario 路径差异:
    #   其它: files/{scenario}/query/{system}/Q{id}.*    (大写 Q)
    #   ecomm: files/ecomm/queries/dialects/{system}/q{id}.*  (复数 + 小写 q)
    # 这是 *声明式 TOML 模式* 的目录约定. 注释 surface 这点, 不修.
    def get_query_text(self, query_id: int, system_name: str) -> str:
        """
        Get the SQL query text for a given query ID and system name.

        Args:
            query_id: ID of the query
            system_name: Name of the system (e.g., "bigquery", "snowflake")
        Returns:
            SQL query text as a string
        """
        system_query_dir = os.path.abspath(
            os.path.join(ECOMM_FILES_DIR, "queries", "dialects", system_name)
        )
        matching_files = glob.glob(
            os.path.join(system_query_dir, f"q{query_id}.*")
        )
        if not matching_files:
            raise FileNotFoundError(
                f"No query implementation found for query {query_id} and system '{system_name}'"
            )
        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple query files found for query {query_id} and system '{system_name}': {matching_files}"
            )

        with open(matching_files[0], "r") as f:
            return f.read()

    # ========================================================================
    # get_ground_truth — ★ ecomm 独有: 在 scenario 里直接跑 DuckDB 算 GT
    # ========================================================================
    # 流程:
    #   1. 从 self.queries[id].definition.ground_truth 拿 SQL (在 TOML 里
    #      预定义好的 *标量* / *deterministic* SQL, 用 DuckDB 可执行的 dialect).
    #   2. duckdb.connect() 开内存 DB.
    #   3. set file_search_path = '<data_dir>' — 告诉 DuckDB 在哪找 CSV /
    #      parquet (允许 SQL 用 `SELECT * FROM 'products.csv'` 这种相对
    #      路径写法).
    #   4. .execute(sql).df() 跑 SQL, 转 pandas DataFrame.
    #   5. con.close() 关连接 (释放 in-memory state).
    #   6. 落盘到 files/ecomm/raw_results/ground_truth/Q{id}.csv 作 cache,
    #      下次 evaluator 直接读 CSV 不用重跑.
    #
    # 为什么用 DuckDB?
    #   - 跨 dialect / 跨 system 的 *中立 baseline*: 不依赖任何 SQPE 的
    #     LLM, 用 deterministic SQL 算出 *客观* 正确答案.
    #   - 嵌入式: 不需要起服务, 跑完释放.
    #   - 支持 CSV / Parquet 直接 query, 不用先 import.
    #
    # 返回 pandas DataFrame (调用方可与各 system output 比较算 metric).
    def get_ground_truth(self, query_id: int) -> pd.DataFrame:
        """
        Get the ground truth result for a given query ID.

        Args:
            query_id: ID of the query
        Returns:
            DataFrame containing the ground truth result
        """
        ground_truth_sql = self.queries[int(query_id)]["definition"][
            "ground_truth"
        ]

        con = duckdb.connect()
        con.execute(f"set file_search_path = '{self.get_data_dir()}'")
        result_df = con.execute(ground_truth_sql).df()
        con.close()

        # Save ground truth CSV
        gt_path = os.path.join(
            ECOMM_FILES_DIR, "raw_results", "ground_truth", f"Q{query_id}.csv"
        )
        os.makedirs(os.path.dirname(gt_path), exist_ok=True)
        result_df.to_csv(gt_path, index=False)

        return result_df

    # ========================================================================
    # get_accuracy_measure_for_query — ★ ecomm 独有: 返回 query 的 accuracy
    #                                  metric 名 (string)
    # ========================================================================
    # 例: query 1 是 selection (sem_filter) → "f1-score"; query 5 是 ranking
    # → "rank_correlation". evaluator 从这里读, 决定调哪个 metric 函数.
    # 这种 *declarative metric spec* 比 evaluator 里 hard-code if-elif 更易
    # 维护; 加新 query 只需在 TOML 写 accuracy_metric, evaluator 不动.
    def get_accuracy_measure_for_query(self, query_id: int) -> str:
        """
        Get the accuracy measure for a given query ID.

        Args:
            query_id: ID of the query

        Returns:
            String representing the accuracy measure (e.g., "precision", "f1-score", etc.)
        """
        try:
            return self.queries[int(query_id)]["definition"]["accuracy_metric"]
        except KeyError as e:
            raise KeyError(f"Missing key {e} for query ID {query_id}.")
