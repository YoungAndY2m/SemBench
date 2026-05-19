"""
============================================================================
教学注释 pass (CLAUDE.md §5.5 全面注释) — animals_scenario.py
============================================================================

Animals scenario 在 SemBench 里的角色
-------------------------------------
跨 *audio + image* 双 modality 的 semantic query 场景. 数据由两张 csv 表
组成:
  - audio_data.csv  音频片段 + 元数据 (e.g. 鸟叫声 / 鲸鱼声)
  - image_data.csv  图片 + 元数据 (e.g. 动物物种 / 栖息地)
query 需要跨两表做 sem_join 等. 默认 scale_factor=500 (实际 audio 上限 650
条, image 上限 8718 条).

Scenario Handler 抽象 (4 方法, 与 mmqa 同) 见 cars/cars_scenario.py:1-50.

Animals 特殊性
--------------
- 数据生成最复杂的 scenario — 不只是简单 download + 切片, 还要保证 query
  特定的 *cross-table pattern*:
    _ensure_cooccurrence_patterns()  保证 audio.animal 与 image.animal
                                      有重叠, 否则 sem_join Q1-Q5 ground truth
                                      全空.
    _ensure_q9_pattern()              特别为 Q9 (audio-image cross 推理)
                                      构造数据.
    _ensure_q6_pattern()              特别为 Q6 (negation query) 构造.
  这些 helper 在 preparation/generate_data.py 里, 类似 stress-test 的
  *adversarial data crafting*.
- 数据 download 从 Google Drive (`download_from_google_drive()`) — 因为
  audio + image 体积大 (源 ~150 MB), 不放 GitHub. 文件 cached 在
  ANIMALS_FILES_DIR/raw/.
- random.seed(42) 让数据生成可复现 (后续 ground truth 也 deterministic).
- 支持 system: bigquery / lotus / palimpzest / thalamusdb (4 个; 没接
  flockmtl 因为 DuckDB 处理音频文件需要额外 extension).

引用: [LOG_STRUCTURE.md §5.3 Scenario 层 — animals](../../../LOG_STRUCTURE.md)
============================================================================
"""

import os
from typing import List
import glob

from scenario.animals.preparation.generate_data import download_from_google_drive


ANIMALS_FILES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "files", "animals")
)


# ============================================================================
# AnimalsScenario — animals use case 的 ScenarioHandler (4 个方法)
# ============================================================================
# 字段:
#   data_dir       初始 None, setup_scenario 跑完才设, 路径形如
#                  files/animals/data/sf_500/
#   scale_factor   image 表的目标行数 (audio 表为 scale_factor // 3, 上限 650)
class AnimalsScenario:
    """
    Animals scenario handler.

    This class:
     * downloads and prepares data
     * retrieves queries
    """

    def __init__(self, scale_factor: int = 500):
        self.data_dir = None
        self.scale_factor = scale_factor

    # ========================================================================
    # setup_scenario — animals 的数据生成最复杂; 4 个 pattern-ensure 阶段
    # ========================================================================
    # 流程:
    #   1. random.seed(42) 让生成可复现.
    #   2. 若 audio_data.csv + image_data.csv 已存在 → skip 生成 (idempotent).
    #   3. 否则 download_from_google_drive() 下载 raw audio + raw image archive.
    #   4. 计算 table size: audio = min(sf//3, 650), image = min(sf, 8718).
    #      audio 比 image 小 ~3x 因为源数据 audio 总量少.
    #   5. _generate_audio_table / _generate_image_table — 从 raw 文件 + 文件
    #      名解析 metadata, 生成 DataFrame.
    #   6. _ensure_cooccurrence_patterns — 保证两表 animal 列有 overlap (Q1-Q5
    #      sem_join 才有非空 ground truth).
    #   7. _ensure_q9_pattern / _ensure_q6_pattern — query-specific pattern.
    #   8. to_csv 落盘.
    #   9. 灌入 systems — bigquery 需要 setup, 其它 system 直接读 csv (no-op).
    #
    # 整个 setup 阶段是 *adversarial data crafting* — 不只是均匀采样, 还要
    # 主动保证某些 query 有有意义的 ground truth, 否则 evaluator 算 P/R/F1
    # 时会出现 division-by-zero.
    def setup_scenario(self, systems: List[str]) -> None:
        # Download and prepare data if not already done
        from scenario.animals.preparation.generate_data import (
            _generate_audio_table,
            _generate_image_table,
            _ensure_cooccurrence_patterns,
            _ensure_q9_pattern,
            _ensure_q6_pattern,
        )
        from pathlib import Path
        import random

        # Set random seed for reproducibility
        random.seed(42)

        # Check if data already exists
        data_folder = Path(ANIMALS_FILES_DIR) / "data" / f"sf_{self.scale_factor}"
        audio_file = data_folder / "audio_data.csv"
        image_file = data_folder / "image_data.csv"

        if audio_file.exists() and image_file.exists():
            print(f"Data already exists at {data_folder}, skipping generation.")
            self.data_dir = str(data_folder)
        else:
            # Download source data
            audio_path, image_path = download_from_google_drive()

            # Calculate table sizes
            max_audio_files = 650
            max_image_files = 8718
            audio_size = min(self.scale_factor // 3, max_audio_files)
            image_size = min(self.scale_factor, max_image_files)

            print(f"Generating tables: Audio={audio_size}, Image={image_size}")

            # Generate tables
            audio_table = _generate_audio_table(audio_path, audio_size)
            image_table = _generate_image_table(image_path, image_size)

            # Ensure co-occurrence patterns
            audio_table, image_table = _ensure_cooccurrence_patterns(
                audio_table, image_table
            )
            audio_table, image_table = _ensure_q9_pattern(audio_table, image_table)
            audio_table, image_table = _ensure_q6_pattern(audio_table, image_table)

            # Save to data directory
            os.makedirs(data_folder, exist_ok=True)
            audio_table.to_csv(audio_file, index=False)
            image_table.to_csv(image_file, index=False)

            print(f"Data saved to {data_folder}")
            self.data_dir = str(data_folder)

        # Load data into the specified systems
        for system in systems:
            if system == "bigquery":
                from scenario.animals.setup.bigquery import BigQueryAnimalsSetup
                from pathlib import Path

                setup = BigQueryAnimalsSetup()
                setup.setup_data(
                    scale_factor=self.scale_factor,
                    data_dir=Path(ANIMALS_FILES_DIR) / "data"
                )
            elif system == "lotus":
                pass  # Nothing to do. LOTUS works on raw files.
            elif system == "palimpzest":
                pass  # Nothing to do. Palimpzest works on raw files.
            elif system == "thalamusdb":
                pass  # Nothing to do. ThalamusDB works on raw files.
            else:
                raise ValueError(f"Unsupported system: {system}")

    # ========================================================================
    # get_query_text — 与 cars / medical 同, 大写 Q{id}.* glob
    # ========================================================================
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
            os.path.join(ANIMALS_FILES_DIR, "query", system_name)
        )
        matching_files = glob.glob(os.path.join(system_query_dir, f"Q{query_id}.*"))

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
    # get_data_dir — 同 mmqa, lazy fallback 到 sf_{N}/ 路径
    # ========================================================================
    def get_data_dir(self) -> str:
        if self.data_dir:
            return self.data_dir
        else:
            return os.path.abspath(
                os.path.join(ANIMALS_FILES_DIR, "data", f"sf_{self.scale_factor}")
            )

    # ========================================================================
    # discover_available_queries — 与 mmqa 同, glob Q*.* 枚举 query ID
    # ========================================================================
    def discover_available_queries(self, system_name: str = None) -> List[int]:
        """
        Discover available queries for the animals scenario.

        Returns:
            List of query IDs.
        """
        if system_name is None:
            # Return all possible queries
            system_query_dir = os.path.abspath(
                os.path.join(ANIMALS_FILES_DIR, "query", "lotus")
            )
        else:
            system_query_dir = os.path.abspath(
                os.path.join(ANIMALS_FILES_DIR, "query", system_name)
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
