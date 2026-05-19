"""
============================================================================
教学注释 pass (CLAUDE.md §5.5 全面注释) — mmqa_scenario.py
============================================================================

MMQA scenario 在 SemBench 里的角色
-----------------------------------
MMQA = Multi-Modal Question Answering. 与其他 scenario 不同, mmqa 的数据
包含 *图像 + 文本 + 表格* 混合 modality (典型场景: 一行带 product
description + product image + price column), query 需要跨 modality 推理.

scale_factor 默认 *25* — 比 medical (11K) 还小 2 个数量级, 因为 MMQA query
计算量大 (每行可能跑 vision LLM + text LLM 各一次), 跑 1 K row 都很贵.

Scenario Handler 抽象 (3 / 4 方法) 见 cars/cars_scenario.py:1-50.

MMQA 特殊性
-----------
- data_dir 在 __init__ 里设 None, *延迟* 到 setup_scenario 里赋值 — 因为
  目录名带 scale_factor (data/sf_{scale_factor}/), 必须先把 generator 跑
  完才知道具体 path.
- 用 MMQADataGenerator (class) 替代其它 scenario 用的 prepare_data
  (function). 因为 mmqa 数据生成步骤多 (下载 → 图片预处理 → ground truth
  标注), 用 class 持有 working_dir / output_data_dir 等中间状态更直观.
- Query 文件名是 *小写* q{id}.* (line 82), 与 medical/cars 的大写 Q{id}.*
  不一致 — 但 discover_available_queries 里又 glob Q*.* (line 124). 这是
  实际 bug, 注释只 surface 不修 ([§D Rule 1]). 实际跑时 Q1 / q1 都能命中,
  因为 macOS HFS+ 和 Linux ext4 默认 case-sensitive, 推测真实 query 文件
  双写都有.
- 多一个方法 discover_available_queries (其它 scenario 没有) — 用于让
  runner 在 user 没指定 --queries 时自动枚举出所有 query.
- setup 里 FlockMTLMMQASetup 缺括号 (line 58: setup = FlockMTLMMQASetup
  而非 FlockMTLMMQASetup()). 实际 setup.setup_data() 当 classmethod 调
  的话需要 cls argument — 这里也只 surface, 由用户上游报告修.

引用: [LOG_STRUCTURE.md §5.3 Scenario 层](../../../LOG_STRUCTURE.md)
============================================================================
"""

import os
from typing import List
import glob


MMQA_FILES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "files", "mmqa")
)


# ============================================================================
# MMQAScenario — mmqa use case 的 ScenarioHandler (4 个方法, 比其它多一个)
# ============================================================================
class MMQAScenario:
    """
    MMQA scenario handler.

    This class:
     * downloads and prepares data
     * retrieves queries
    """

    def __init__(self, scale_factor: int = 25):
        self.data_dir = None
        self.scale_factor = scale_factor

    # ========================================================================
    # setup_scenario — 准备数据 + 按 system 灌入 storage
    # ========================================================================
    # 与其它 scenario 不同之处:
    #   1. *先查 data_folder/{sf_N}/*.csv 是否已存在*: 存在则跳过 generation
    #      (idempotent + 手动加速; 主要 cost 是图片预处理).
    #   2. 用 MMQADataGenerator class (持有 working_dir / output_data_dir /
    #      skip_download 等 state) 而不是简单函数. generate_data() 内部:
    #         a. download raw multi-modal corpus
    #         b. crop / resize image (~25 examples)
    #         c. 生成 query-specific ground truth
    #      最终把 self.data_dir 设成 generator.output_data_dir.
    #   3. system 支持: bigquery / flockmtl / lotus / palimpzest / thalamusdb
    #      (5 个; 没接 snowflake 也没接 caesura).
    def setup_scenario(self, systems: List[str]) -> None:
        # Download and prepare data if not already done
        from scenario.mmqa.preparation.generate_data import MMQADataGenerator
        from pathlib import Path

        # Get the working directory (base directory of the project)
        working_dir = Path(__file__).resolve().parents[3]

        # Check if data already exists
        data_folder = Path(MMQA_FILES_DIR) / "data" / f"sf_{self.scale_factor}"

        if data_folder.exists() and len(list(data_folder.glob("*.csv"))) > 0:
            print(f"Data already exists at {data_folder}, skipping generation.")
            self.data_dir = str(data_folder)
        else:
            # Generate data
            data_generator = MMQADataGenerator(
                working_dir=str(working_dir),
                scale_factor=self.scale_factor,
                skip_download=False,
            )
            data_generator.generate_data()
            self.data_dir = data_generator.output_data_dir

        # Load data into the specified systems
        for system in systems:
            if system == "bigquery":
                from scenario.mmqa.setup.bigquery import BigQueryMMQASetup

                setup = BigQueryMMQASetup()
                setup.setup_data(self.data_dir)
            elif system == "flockmtl":
                from scenario.mmqa.setup.flockmtl import FlockMTLMMQASetup
                
                setup = FlockMTLMMQASetup
                setup.setup_data(self.data_dir)
            elif system == "lotus":
                pass  # Nothing to do. LOTUS works on raw files.
            elif system == "palimpzest":
                pass  # Nothing to do. Palimpzest works on raw files.
            elif system == "thalamusdb":
                pass  # Nothing to do. ThalamusDB works on raw files.
            else:
                raise ValueError(f"Unsupported system: {system}")

    # ========================================================================
    # get_query_text — 从磁盘读 query 文件; 这里 glob 用 *小写* q{id}.*
    # ========================================================================
    # 与 medical / cars 的 Q{id}.* (大写) 不一致, 见顶部 docstring 的 surface
    # 说明. 不修, 仅记录.
    def get_query_text(self, query_id, system_name: str) -> str:
        """
        Get the SQL query text for a given query ID and system name.

        Args:
            query_id: ID of the query
            system_name: Name of the system
        Returns:
            SQL query text as a string
        """
        system_query_dir = os.path.abspath(
            os.path.join(MMQA_FILES_DIR, "query", system_name)
        )
        matching_files = glob.glob(os.path.join(system_query_dir, f"q{query_id}.*"))

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
    # get_data_dir — 返回 mmqa 本地数据目录; data_dir 未设时拼默认 sf_{N} 路径
    # ========================================================================
    # 与其它 scenario 不同, self.data_dir 不一定在 __init__ 里就设好
    # (e.g. --skip-setup 时根本没跑 setup_scenario, self.data_dir 仍为 None);
    # 这种情况下 fallback 到 files/mmqa/data/sf_{scale_factor}/ 推测路径.
    def get_data_dir(self) -> str:
        if self.data_dir:
            return self.data_dir
        else:
            return os.path.abspath(
                os.path.join(MMQA_FILES_DIR, "data", f"sf_{self.scale_factor}")
            )

    # ========================================================================
    # discover_available_queries — MMQA 独有: 枚举所有可用 query ID
    # ========================================================================
    # 与 cars / medical 不同 — 它们的 query ID 由 runner 自己用 hard-coded
    # range 决定; mmqa 因为 query 列表可能动态 (不同 scale_factor 下可用
    # query 不一样), 提供发现机制. 调用方:
    #   runner.run_all_queries(queries=None) 时, runner 看到 None 会调本方法
    #   拿到全集.
    #
    # 实现:
    #   1. 取 system_name 对应 query 目录 (默认 lotus, 即用 lotus 的 query
    #      集做参考 — 假设各 system query 集合相同).
    #   2. glob Q*.* 把所有 query 文件枚举出来. 注意这里又是 *大写* Q
    #      (跟 get_query_text 的 *小写* q 不一致, surface 标记).
    #   3. 文件名 strip 头 "Q" + 取 "." 前的数字 → int.
    #   4. 解析失败 (e.g. 文件名不是 Q\d+ 格式) 直接 skip.
    #   5. sorted 返回升序 ID 列表.
    def discover_available_queries(self, system_name: str = None) -> List[int]:
        """
        Discover available queries for the MMQA scenario.

        Returns:
            List of query IDs.
        """
        if system_name is None:
            # Return all possible queries from lotus directory
            system_query_dir = os.path.abspath(
                os.path.join(MMQA_FILES_DIR, "query", "lotus")
            )
        else:
            system_query_dir = os.path.abspath(
                os.path.join(MMQA_FILES_DIR, "query", system_name)
            )

        if not os.path.exists(system_query_dir):
            return []

        query_files = glob.glob(os.path.join(system_query_dir, "Q*.*"))
        query_ids = []
        for f in query_files:
            try:
                # Extract number from filename like Q1.py or Q1.sql
                filename = os.path.basename(f)
                query_id = int(filename[1:].split(".")[0])
                query_ids.append(query_id)
            except ValueError:
                continue
        return sorted(query_ids)
