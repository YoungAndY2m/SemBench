"""
============================================================================
教学注释 pass (CLAUDE.md §5.5 全面注释) — medical_scenario.py
============================================================================

Medical scenario 在 SemBench 里的角色
-------------------------------------
模拟医疗领域 (临床记录 + 诊断) 上的语义查询. scale_factor 默认 11,112 行 —
比 cars (157K) / ecomm (44K) 小一个数量级, 因为医疗 ground truth 标注成本
高 (需要医学专业知识).

Scenario Handler 的抽象 + 三大方法的通用语义已在 cars_scenario.py 顶部
教学注释里展开, 本文件只标注 *medical 特有* 的点 (snowflake / flockmtl /
thalamusdb 各自的 setup 形态). 通用部分参考:
  [src/scenario/cars/cars_scenario.py:1-50](../cars/cars_scenario.py#L1)

Medical 特殊性
--------------
- 唯一一个接 snowflake 的 scenario (cars 不接, 因为 cars 没 snowflake setup
  脚本). Snowflake 走 enterprise data warehouse 通路, 与 BigQuery 平行.
- thalamusdb 走 'pass' (no-op) — 在 medical 场景下 ThalamusDB 也直接读 raw
  CSV, 不需要预 load. 与 cars 不同 (cars 的 thalamusdb 要预 load 到 DuckDB).
- 支持的 system: bigquery / snowflake / flockmtl / lotus / palimpzest /
  thalamusdb (6 个, 最多).

引用: [LOG_STRUCTURE.md §5.3 Scenario 层](../../../LOG_STRUCTURE.md)
============================================================================
"""

import os
from typing import List

from scenario.medical.preparation.generate_data import prepare_data
import glob


MEDICAL_FILES_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "files", "medical"
    )
)


# ============================================================================
# MedicalScenario — medical use case 的 ScenarioHandler
# ============================================================================
# 字段:
#   data_dir       本地数据目录 (默认 <repo_root>/files/medical/)
#   scale_factor   生成数据集行数 (默认 11,112; 范围在 generate_data.py 里
#                  由配置文件决定)
class MedicalScenario:
    """
    Medical scenario handler.

    This class:
     * downloads and prepares data
     * retrievs queries
    """

    def __init__(self, scale_factor: int = 11112):
        self.data_dir = MEDICAL_FILES_DIR
        self.scale_factor = scale_factor

    # ========================================================================
    # setup_scenario — 准备数据 + 按 system 灌入 storage
    # ========================================================================
    # 通用模式见 cars/cars_scenario.py 顶层注释; medical 特有的 system
    # mapping:
    #   bigquery / snowflake / flockmtl  → 上传到对应 warehouse / DuckDB
    #   lotus / palimpzest / thalamusdb  → no-op (直接读 raw 文件)
    #
    # Snowflake 的 BigQueryMedicalSetup-equivalent 用 snowflake-connector-python
    # SDK, 把 CSV 用 PUT + COPY INTO 推到 Snowflake stage 再灌进 table.
    # 见 [src/scenario/medical/setup/bigquery.py] 和 [snowflake.py].
    def setup_scenario(self, systems: List[str]) -> None:
        # Download and prepare data if not already done
        prepare_data(scaling_factor=self.scale_factor)

        # Load data into the specified systems
        for system in systems:
            if system == "bigquery":
                from scenario.medical.setup.bigquery import BigQueryMedicalSetup

                setup = BigQueryMedicalSetup()
                setup.setup_data(data_dir=self.data_dir, scale_factor=self.scale_factor)

            elif system == "snowflake":
                from scenario.medical.setup.snowflake import (
                    SnowflakeMedicalSetup,
                )

                setup = SnowflakeMedicalSetup()
                setup.setup_data(self.data_dir, self.scale_factor)
            
            elif system == "flockmtl":
                from scenario.medical.setup.flockmtl import FlockMTLMedicalSetup

                setup = FlockMTLMedicalSetup()
                setup.setup_data(self.data_dir, self.scale_factor)
                
            elif system == "lotus":
                pass  # Nothing to do. LOTUS works on raw files.

            elif system == "palimpzest":
                pass  # Nothing to do. Palimpzest works on raw files.
            
            elif system == "thalamusdb":
                pass

            else:
                raise ValueError(f"Unsupported system: {system}")

    # ========================================================================
    # get_query_text — 从磁盘 glob 出 query 文件返回文本
    # ========================================================================
    # 通用语义参考 cars/cars_scenario.py:get_query_text. medical 没差别.
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
            os.path.join(MEDICAL_FILES_DIR, "query", system_name)
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
    # get_data_dir — 返回 medical scenario 本地数据目录
    # ========================================================================
    def get_data_dir(self) -> str:
        return self.data_dir
