"""
============================================================================
教学注释 pass (CLAUDE.md §5.5 全面注释) — cars_scenario.py
============================================================================

Scenario Handler 抽象 (本 repo 6 个 scenario 都遵循同一模式)
------------------------------------------------------------
SemBench 的 *scenario* = 一个独立的 use case (cars / movie / ecomm /
medical / animals / mmqa), 包含:
  - 数据集 (raw + 派生表格)
  - 一组 query (放在 files/{scenario}/query/{system}/Q{id}.<ext>)
  - 不同 system (lotus / palimpzest / bigquery / thalamusdb / flockmtl /
    caesura) 需要的 setup 方式 (有的要 load 到 BigQuery, 有的直接读 file).

ScenarioHandler 类是 *per-scenario* 的胶水层. 三个核心方法:
  setup_scenario(systems)   = 准备数据 (下载 + 派生) + 灌入各 system 的存储
  get_query_text(qid, sys)  = 从磁盘读 Q{qid}.<ext> 文件返回 query 文本
                              (ext 因 system 而异: SQL / Python / DuckDB SQL)
  get_data_dir()            = 返回该 scenario 的本地数据目录

调用方: src/runner/generic_runner.py:get_scenario_handler() 在 runner init
时根据 use_case 实例化对应的 ScenarioHandler.

Cars scenario 特殊性
--------------------
- scale_factor 默认 157376 (≈ 15 万行 car listings, 从 Kaggle 派生).
- 支持的 system: bigquery / lotus / palimpzest / thalamusdb / flockmtl
  (没接 caesura).
- LOTUS / Palimpzest setup 是 no-op — 它们直接读 raw CSV / image 文件,
  不需要先建表.

加新 scenario 时改动点 ([LOG_STRUCTURE.md §7.2](../../../LOG_STRUCTURE.md))
  1. 拷贝本目录结构 (src/scenario/cars/) → 新目录
  2. 实现 {Name}Scenario 类 (本类的 5 个方法)
  3. config/ 加 prompt / query / 数据 manifest
  4. src/runner/generic_runner.py:get_scenario_handler 加 elif
  5. src/run.py:get_evaluator 加 mapping
============================================================================
"""

import os
from typing import List

from scenario.cars.preparation.generate_data import prepare_data
import glob


CARS_FILES_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "files", "cars"
    )
)


# ============================================================================
# CarsScenario — cars use case 的 ScenarioHandler
# ============================================================================
# 字段:
#   data_dir       本地数据目录 (默认 <repo_root>/files/cars/)
#   scale_factor   生成数据集的规模 (默认 157,376 行)
#
# 注意: docstring 里写的是 "Medical scenario handler" — 这是 copy-paste 错误
# (从 medical_scenario.py 拷过来的); 按 CLAUDE.md §5.5 §D Rule 1 "只能增加,
# 不能动原始代码", 这里不修. 仅做 surface 记录, 上游 push 时统一报告.
class CarsScenario:
    """
    Medical scenario handler.

    This class:
     * downloads and prepares data
     * retrievs queries
    """

    def __init__(self, scale_factor: int = 157376):
        self.data_dir = CARS_FILES_DIR
        self.scale_factor = scale_factor

    # ========================================================================
    # setup_scenario — 准备数据 + 按 system 分别 setup
    # ========================================================================
    # 两阶段:
    #   1. prepare_data(scaling_factor) 在 preparation/generate_data.py 里
    #      做数据生成 (下载 raw CSV/image + 派生 cars 表 + per-query ground
    #      truth). idempotent (跑过一次再跑会 short-circuit, 由 generate_data
    #      内部判 marker file 决定).
    #   2. for system in systems: 按 system 自家存储 layer 灌数据.
    #      - bigquery → 上传到 GCP BigQuery dataset
    #      - lotus / palimpzest → no-op (它们读 raw 文件)
    #      - thalamusdb → 灌 DuckDB
    #      - flockmtl → 灌 DuckDB + load extension
    #      - 未知 system → ValueError (强 fail loud, 防止漏写 elif)
    #
    # *延迟* import 各 BigQueryCarsSetup / ThalamusDBCarsSetup / ... 是因为
    # 它们各自的 setup module 依赖各自 system 的 SDK (google-cloud-bigquery /
    # duckdb), 在 sembench 主 venv 里没装. 把 import 写在 elif 里, 只在真
    # 跑那个 system 时才 import.
    def setup_scenario(self, systems: List[str]) -> None:
        # Download and prepare data if not already done
        prepare_data(scaling_factor=self.scale_factor)

        # Load data into the specified systems
        for system in systems:
            if system == "bigquery":
                from scenario.cars.setup.bigquery import BigQueryCarsSetup

                setup = BigQueryCarsSetup()
                setup.setup_data(data_dir=os.path.join(self.data_dir, "data"), scale_factor=self.scale_factor)

            elif system == "lotus":
                pass  # Nothing to do. LOTUS works on raw files.

            elif system == "palimpzest":
                pass  # Nothing to do. Palimpzest works on raw files.
            
            elif system == "thalamusdb":
                from scenario.cars.setup.thalamusdb import ThalamusDBCarsSetup

                setup = ThalamusDBCarsSetup()
                setup.setup_data(data_dir=self.data_dir, scale_factor=self.scale_factor)

            elif system == "flockmtl":
                from scenario.cars.setup.flockmtl import FlockMTLCarsSetup

                setup = FlockMTLCarsSetup()
                setup.setup_data(data_dir=self.data_dir, scale_factor=self.scale_factor)

            else:
                raise ValueError(f"Unsupported system: {system}")

    # ========================================================================
    # get_query_text — 从磁盘读 query 文件返回 query string
    # ========================================================================
    # 文件路径约定:
    #   files/cars/query/{system_name}/Q{query_id}.<ext>
    # 不同 system 用不同语法/扩展名:
    #   bigquery   → Q1.sql        (BigQuery SQL with AI.IF / AI.GENERATE)
    #   lotus      → Q1.py         (Python with df.sem_filter / sem_join)
    #   palimpzest → Q1.py         (pz.Dataset + pz.Filter / pz.Convert DSL)
    #   thalamusdb → Q1.sql        (DuckDB SQL + semantic predicate)
    #   flockmtl   → Q1.sql        (DuckDB SQL + flockmtl scalar UDF)
    # 用 glob 匹配 Q{id}.* 而不是硬指定 ext, 给加 system 留口子.
    #
    # 防御:
    #   - 0 个匹配 → FileNotFoundError (查 typo / 漏文件)
    #   - 2+ 个匹配 → ValueError (e.g. Q1.sql 和 Q1.py 同时存在, 不确定用哪)
    def get_query_text(self, query_id: int, system_name: str) -> str:
        """
        Get the SQL query text for a given query ID and system name.

        Args:
            query_id: ID of the query
            system_name: Name of the system
        Returns:
            SQL query text as a string
        """
        system_query_dir = os.path.abspath(
            os.path.join(CARS_FILES_DIR, "query", system_name)
        )
        matching_files = glob.glob(
            os.path.join(system_query_dir, f"Q{query_id}.*")
        )

        if not matching_files:
            print(os.path.join(system_query_dir, f"Q{query_id}.*"))
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
    # get_data_dir — 返回该 scenario 本地数据目录
    # ========================================================================
    # 各 runner 拿到 data_dir 后从中读 raw CSV / parquet / image. 通常是
    # <repo_root>/files/cars/ (CARS_FILES_DIR 模块级常量).
    def get_data_dir(self) -> str:
        return self.data_dir
