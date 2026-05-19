"""
============================================================================
教学注释 pass (CLAUDE.md §5.5 全面注释) — movie_scenario.py
============================================================================

Movie scenario 在 SemBench 里的角色
-----------------------------------
电影 + 影评 数据集 (Movies.csv + Reviews.csv) 上的 semantic query. 默认
scale_factor=2000 评论 (从 Amazon Movies & TV reviews 派生, source 在
Google Drive). 是 SemBench 论文里跑得最完整的 scenario, 6 个 system 全
接 (含 caesura).

Scenario Handler 抽象 (4 方法, 与 mmqa / animals 同) 见
cars/cars_scenario.py:1-50.

Movie 特殊性
------------
- 数据生成是 6 个 scenario 里最 *混合采样* 的:
    find_pattern_movie     找有特定模式的电影 (e.g. 影评里出现 "absolutely"
                           的电影), 用来测 sem_filter Q1.
    get_negative_movie     hard-coded 一个差评最多的电影, 用来测 sem_join
                           或 negation query.
    get_top_movies_fast    按 review count 排序取 top-N, 用来测 ranking
                           query.
    sample_reviews         按上面三类 + 普通 review 拼出 scale_factor 条评论.
    generate_movies_table  从 reviews 反推出现的电影集合, 拼 Movies.csv.
- reviewText 在落盘前要 strip 换行 (line 76-81) — 因为 CSV reader 默认
  会把换行当行分隔符, review 里的换行会破坏 CSV format. 这是 *踩过坑后
  补的 workaround*, 不是 design.
- 唯一接 caesura 的 scenario (其它 scenario 没有 caesura runner) —
  caesura paper 的 evaluation 就用 movie scenario.
- 支持 system: 6 个 (bigquery / flockmtl / lotus / caesura / palimpzest /
  thalamusdb).

引用: [LOG_STRUCTURE.md §5.3 Scenario 层 — movie](../../../LOG_STRUCTURE.md)
============================================================================
"""

import os
from typing import List
import glob


MOVIE_FILES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "files", "movie")
)


# ============================================================================
# MovieScenario — movie use case 的 ScenarioHandler (4 方法)
# ============================================================================
# 字段:
#   data_dir       初始 None, setup 跑完才设, 路径 files/movie/data/sf_2000/
#   scale_factor   sample 出的 review 条数 (Movies.csv 的行数由 review 推出)
class MovieScenario:
    """
    Movie scenario handler.

    This class:
     * downloads and prepares data
     * retrieves queries
    """

    def __init__(self, scale_factor: int = 2000):
        self.data_dir = None
        self.scale_factor = scale_factor

    # ========================================================================
    # setup_scenario — 混合采样生成 Movies.csv + Reviews.csv → 灌入各 system
    # ========================================================================
    # 详细流程见顶部 docstring "Movie 特殊性" 段; 关键采样函数:
    #   find_pattern_movie  → Q1 用 (sem_filter on pattern)
    #   get_negative_movie  → Q5 / Q7 用 (negation / sentiment)
    #   get_top_movies_fast → Q3 用 (rank / aggregation)
    #   sample_reviews      → 把 4 类 review 拼 scale_factor 条
    # 然后 generate_movies_table 反推 Movies.csv 表.
    #
    # 落盘前 reviewText.str.replace("\n", " ") — 是 *CSV format-safety* 的
    # workaround, 不要改成 to_csv 的 quoting 参数 (那样会改变文件格式 →
    # 影响下游 system load).
    def setup_scenario(self, systems: List[str]) -> None:
        # Download and prepare data if not already done
        from scenario.movie.preparation.generate_data import (
            download_from_google_drive,
            load_data,
            find_pattern_movie,
            get_negative_movie,
            get_top_movies_fast,
            sample_reviews,
            generate_movies_table,
        )
        from pathlib import Path

        # Check if data already exists
        data_folder = Path(MOVIE_FILES_DIR) / "data" / f"sf_{self.scale_factor}"
        movies_file = data_folder / "Movies.csv"
        reviews_file = data_folder / "Reviews.csv"

        if movies_file.exists() and reviews_file.exists():
            print(f"Data already exists at {data_folder}, skipping generation.")
            self.data_dir = str(data_folder)
        else:
            # Download source data
            data_path = download_from_google_drive()

            # Load data
            movies_df, reviews_df = load_data(data_path)

            # Find special movies
            pattern_movie, pattern_pattern = find_pattern_movie(reviews_df)
            negative_movie = get_negative_movie()

            # Get top movies
            top_movies = get_top_movies_fast(reviews_df)

            # Sample reviews
            selected_reviews = sample_reviews(
                reviews_df,
                pattern_movie,
                pattern_pattern,
                negative_movie,
                top_movies,
                self.scale_factor,
            )

            # Generate movies table
            selected_movies = generate_movies_table(movies_df, selected_reviews)

            # Save to data directory
            os.makedirs(data_folder, exist_ok=True)

            # Clean reviewText to prevent CSV formatting issues
            selected_reviews = selected_reviews.copy()
            selected_reviews["reviewText"] = (
                selected_reviews["reviewText"]
                .str.replace("\n", " ", regex=False)
                .str.replace("\r", " ", regex=False)
            )

            selected_movies.to_csv(movies_file, index=False)
            selected_reviews.to_csv(reviews_file, index=False)

            print(f"Data saved to {data_folder}")
            self.data_dir = str(data_folder)

        # Load data into the specified systems
        for system in systems:
            if system == "bigquery":
                from scenario.movie.setup.bigquery import BigQueryMovieSetup
                from pathlib import Path

                setup = BigQueryMovieSetup()
                setup.setup_data(
                    scale_factor=self.scale_factor,
                    data_dir=Path(MOVIE_FILES_DIR) / "data"
                )
            elif system == "flockmtl":
                from scenario.movie.setup.flockmtl import FlockMTLMovieSetup

                setup = FlockMTLMovieSetup()
                setup.setup_data()
            elif system == "lotus":
                pass  # Nothing to do. LOTUS works on raw files.
            elif system == "caesura":
                pass  # Nothing to do. Caesura works on raw files.
            elif system == "palimpzest":
                pass  # Nothing to do. Palimpzest works on raw files.
            elif system == "thalamusdb":
                pass  # Nothing to do. ThalamusDB works on raw files.
            else:
                raise ValueError(f"Unsupported system: {system}")

    # ========================================================================
    # get_query_text — 与 cars / animals 同, 大写 Q{id}.* glob
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
            os.path.join(MOVIE_FILES_DIR, "query", system_name)
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
    # get_data_dir / discover_available_queries — 与 mmqa / animals 同义
    # ========================================================================
    def get_data_dir(self) -> str:
        if self.data_dir:
            return self.data_dir
        else:
            return os.path.abspath(
                os.path.join(MOVIE_FILES_DIR, "data", f"sf_{self.scale_factor}")
            )

    def discover_available_queries(self, system_name: str = None) -> List[int]:
        """
        Discover available queries for the movie scenario.

        Returns:
            List of query IDs.
        """
        if system_name is None:
            # Return all possible queries from lotus directory
            system_query_dir = os.path.abspath(
                os.path.join(MOVIE_FILES_DIR, "query", "lotus")
            )
        else:
            system_query_dir = os.path.abspath(
                os.path.join(MOVIE_FILES_DIR, "query", system_name)
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
