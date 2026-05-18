"""
Created on August 1, 2025

@author: Jiale Lao

CAESURA runner implementation for movie use case.

============================================================================
SemBench/src/scenario/movie/runner/caesura_runner/caesura_runner.py
SemBench L1 wrapper — movie scenario 专属 CAESURA runner
============================================================================

教学注释 pass (L1 ADD) by Claude.
对应 [CAESURA LOG_STRUCTURE.md §10.1.1 L1 ADD 表](../../../../../../AllSQPE/CAESURA/LOG_STRUCTURE.md#1011-l1-add--新增文件l0-没有的)
中的 `movie/runner/caesura_runner.py` 行 (468 LoC)。

----------------------------------------------------------------------------
这个文件干什么 (一句话)
----------------------------------------------------------------------------
继承 `GenericCaesuraRunner`, 专门为 movie scenario 提供:
1. **数据清洗的 `_setup_database_from_files()` 覆盖**:
   - drop 列: originalScore / scoreSentiment / reviewState
   - drop 行: reviewText 为 null 的
   - 重排列序: reviewText 移到最后一列 (CAESURA TEXT 列约定)
2. **8 个 query 的 `_execute_q1..q8` 实现**:
   每个 query 调 `execute_caesura_query(NL string)` 拿 DataFrame, 再 *按 evaluator 期望 schema* 提列/重命名。
3. **`_get_empty_results_dataframe(query_id)` 覆盖**: 给每个 q_id 返回 *带正确列名* 的空 df, failed query 走这个 fallback。

----------------------------------------------------------------------------
movie scenario 的 8 个 query 类型
----------------------------------------------------------------------------
| Q  | 类型           | 期望 schema                        | 例子                          |
|----|----------------|------------------------------------|-------------------------------|
| Q1 | top-K          | reviewId                           | 5 clearly positive reviews    |
| Q2 | top-K filter   | reviewId                           | 5 positive for taken_3        |
| Q3 | count          | count                              | count positive for taken_3    |
| Q4 | ratio/avg      | positivity_ratio                   | positivity ratio for taken_3  |
| Q5 | self-join pair | id, reviewId1, reviewId2 (10 行)   | 10 pairs same sentiment       |
| Q6 | self-join pair | id, reviewId1, reviewId2 (10 行)   | 10 pairs opposite sentiment   |
| Q7 | self-join pair | id, reviewId1, reviewId2 (全部)    | all pairs opposite sentiment  |
| Q8 | group-by count | scoreSentiment, count              | sentiment counts for taken_3  |

----------------------------------------------------------------------------
schema-adapter 模式 (本文件的核心抽象)
----------------------------------------------------------------------------
LLM 输出的 DataFrame *列名* 是 LLM 自由发挥的 (e.g. "review_id" / "ReviewID" / "id"),
但 evaluator 期望固定列名 (e.g. "reviewId"); 所以每个 _execute_q*N* 都做"列查找 + 重命名":
  1. 用关键词模糊匹配找 "包含 reviewId / count / sentiment" 的列
  2. fallback: 直接用第 N 列 (位置匹配)
  3. .rename + .head(K) 输出标准化结果

这种 "LLM 输出 → 期望 schema" 的 adapter 是 SemBench evaluator 工作的关键 —
否则 LLM 列名随机, 评测无从对齐。

----------------------------------------------------------------------------
零基础读者预备知识 (Python + pandas)
----------------------------------------------------------------------------
- `Path(__file__).parent.parent.parent.parent` (4 次 .parent):
    `__file__` 是当前文件路径; .parent 一次跳一级目录;
    4 次跳到 SemBench/src/, 5 次跳到 SemBench/。
    用 .parent 链是为了 sys.path 注入"相对 import 不能直接处理"的远祖目录。

- `df.drop(columns=[...])`: pandas 删列, 返回新 df (不 in-place);
- `df.dropna(subset=["col"])`: 删 col 是 NaN 的行;
- `df.rename(columns={old: new})`: 改列名, 返回新 df;
- `df.select_dtypes(include=["number"])`: 按 dtype 筛列 — 这里取数值列, 用于 fallback 找 "count/ratio";
- `[col for col in cols if "reviewid" in col.lower()]`: list comprehension + 大小写不敏感的子串匹配。

- `self.get_query_text(query_id, "natural_language")` (继承自 GenericRunner):
    去 SemBench 标准目录 files/movie/query/natural_language/Q<id>.txt 读 NL query 文本;
    CAESURA 与其它 system 不同 — 不读 SQL 模板, 直接吃 NL string。

----------------------------------------------------------------------------
对应 paper / 配套
----------------------------------------------------------------------------
- 评测 schema 期望见 SemBench evaluator (见 [SemBench/LOG_STRUCTURE.md](../../../../../LOG_STRUCTURE.md))
- 与 GenericCaesuraRunner 的合作详 [generic_caesura_runner.py file docstring](../../../../runner/generic_caesura_runner/generic_caesura_runner.py)
- CAESURA L0 整 pipeline 详 [CAESURA LOG_STRUCTURE.md](../../../../../../AllSQPE/CAESURA/LOG_STRUCTURE.md)
"""

import os
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import sys

# Add parent directories to path for imports
# Path(__file__).parent.parent.parent.parent = SemBench/src/  (4 次 .parent 从本文件跳到 src)
# 给 sys.path 加上 SemBench/src — 让后面 `from runner.generic_caesura_runner.*` 能 import
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
# 5 次 .parent = SemBench/ ; 再拼 runner/generic_caesura_runner
# 给 sys.path 加上 generic_caesura_runner/ — 让 vendored caesura/ 包能 import (它自己 __init__ 也做这件事)
generic_caesura_path = str(
    Path(__file__).parent.parent.parent.parent.parent
    / "runner"
    / "generic_caesura_runner"
)
sys.path.insert(0, generic_caesura_path)

# 必须在 sys.path 修改后才能 import — 否则 Python 找不到 runner.generic_caesura_runner 包
from runner.generic_caesura_runner.generic_caesura_runner import (
    GenericCaesuraRunner,
)


class CaesuraRunner(GenericCaesuraRunner):
    """CAESURA runner for movie use case."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker: int = 1,
        skip_setup: bool = False,
    ):
        """
        Initialize CAESURA runner for movie case.

        Args:
            use_case: The use case to run
            model_name: LLM model to use
            concurrent_llm_worker: Number of concurrent workers
            skip_setup: Whether to skip setup
        """
        # 默认 model_name="gemini-2.5-flash" — 这是 SemBench 通用 default;
        # 但传给 GenericCaesuraRunner._map_model_name 时会 fallback 到 "gpt-4-0613"
        # (CAESURA 只认 0613 snapshot — 详 generic_caesura_runner.py 的 _map_model_name 注释)
        # 实际效果: 用户写 "gemini-2.5-flash" 但跑的是 gpt-4-0613 — 静默 fallback
        # ⚠ 注意! 这里参数顺序 (use_case, scale_factor, model_name, ...) 与父类期望一致;
        # 但 GenericCaesuraRunner.__init__ 又把 (scale_factor, model_name) 顺序传错给 GenericRunner.super
        # 详 generic_caesura_runner.py 文件级 docstring 的 BUG 说明
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )

    def _setup_database_from_files(self):
        """
        Setup CAESURA database using movie data files with required data manipulation.

        Data manipulation requirements:
        - Drop originalScore and scoreSentiment columns
        - Move reviewText column to the last column
        - Drop rows that have null value for reviewText
        """
        # =====================================================================
        # 覆盖父类 _setup_database_from_files — 加 3 步数据清洗
        # =====================================================================
        # 为什么要清洗:
        # 1. drop originalScore/scoreSentiment/reviewState — 这些列是 *标准答案* (ground truth label),
        #    必须从输入数据移走, 否则 LLM 会直接抄而不是真正分析文本
        # 2. drop null reviewText 行 — TextQATool 不能处理 NaN; 这些行也无信息可抽
        # 3. 把 reviewText 移到 *最后一列* — CAESURA add_text_table 约定 "最后一列被标 TEXT 类型"
        #    (见 table.py:create_text_table); 不移的话 LLM 不知道该用 TextQATool
        from caesura.database.database import Database

        db = Database()

        # Load movie data from MMBench files
        # ⚠ 文件名 "Movies_2000.csv" / "Reviews_2000.csv" 是 hardcoded; 实际 SemBench 当前 data dir 是 "Movies.csv"/"Reviews.csv"
        # 详 LOG.md "遗留问题 #2"
        movies_path = str(self.data_path / "Movies_2000.csv")
        reviews_path = str(self.data_path / "Reviews_2000.csv")

        # Load and process reviews data
        # 先把 reviews 加载成 pandas DataFrame 做清洗 (CAESURA add_*_table 只接 CSV 路径不接 DataFrame)
        reviews_df = pd.read_csv(reviews_path)

        # Data manipulation for reviews:
        # 1. Drop originalScore and scoreSentiment columns if they exist
        # 这 3 列是 ground truth 标签 — 必须先 drop 防止 LLM 作弊
        columns_to_drop = ["originalScore", "scoreSentiment", "reviewState"]
        for col in columns_to_drop:
            # `if col in df.columns`: 防御性检查 (CSV schema 可能演化, 列不存在时不报错)
            if col in reviews_df.columns:
                reviews_df = reviews_df.drop(columns=[col])

        # 2. Drop rows with null reviewText
        if "reviewText" in reviews_df.columns:
            # dropna(subset=["reviewText"]): 只检查 reviewText 列, 其它列 NaN 保留
            reviews_df = reviews_df.dropna(subset=["reviewText"])

            # 3. Move reviewText column to the last position
            # pandas 没有 in-place "move column to end" API, 手动实现: 取出 → drop → 重新 append
            reviewText_col = reviews_df["reviewText"]
            reviews_df = reviews_df.drop(columns=["reviewText"])
            reviews_df["reviewText"] = reviewText_col

        # Save processed reviews to a temporary file for CAESURA
        # CAESURA add_*_table 只接 CSV 路径 → 必须把清洗后的 df 写盘
        # ⚠ 副作用: 在 data_path 下落了一个 "temp_reviews_processed.csv" 文件; 没清理逻辑
        # 多次跑同 use case 会反复覆盖该文件 (无 cache)
        temp_reviews_path = str(self.data_path / "temp_reviews_processed.csv")
        reviews_df.to_csv(temp_reviews_path, index=False)

        # Add movies table (no manipulation needed)
        # movies 表不清洗 — 不含 LLM 需"看"的长文本; 直接 add_tabular_table
        db.add_tabular_table(
            "movies",
            movies_path,
            "a table that contains information about movies including titles, scores, ratings, genres, directors, and other metadata",
        )

        # Add processed reviews table
        # add_text_table 注册的最后一列 = reviewText (上面已移到末尾) → 自动标 TEXT 类型
        db.add_text_table(
            "reviews",
            temp_reviews_path,
            "a table that contains movie reviews with metadata and review text content for sentiment analysis",
        )

        # Link reviews to movies by movie id
        # 注意 link 是 *单向* 的: 只挂到 reviews.links, 不挂 movies.links
        # Discovery Phase 的 DFS 算法依赖有向边假设找 join key
        db.link("reviews", "movies", "id")

        # Build relevant values index for key categorical columns
        # 给 categorical 列建 n-gram 模糊索引 — Discovery Phase 用 query 关键词模糊匹配回值
        db.build_relevant_values_index(
            "movies", "genre", "director", "rating", "originalLanguage"
        )

        # Build index for reviews table (excluding dropped columns)
        # 防御性: 上面可能 drop 了 reviewState; 这里检查每列是否还存在再建索引
        # ⚠ 注意! reviewState 在 columns_to_drop 里已被 drop, 这里循环只会走 publicationName 和 isTopCritic
        # 看起来逻辑有点冗余 — 但保留 reviewState 在 list 里是为了将来 *不 drop reviewState* 时能自动加回索引
        available_review_cols = []
        for col in ["reviewState", "publicationName", "isTopCritic"]:
            if col in reviews_df.columns:
                available_review_cols.append(col)

        if available_review_cols:
            # *available_review_cols: 把 list 展开成 positional args (e.g. build_relevant_values_index("reviews", "publicationName", "isTopCritic"))
            db.build_relevant_values_index("reviews", *available_review_cols)

        return db

    def _get_empty_results_dataframe(self, query_id: int) -> pd.DataFrame:
        """
        Get empty DataFrame with correct columns for each movie query.

        Args:
            query_id: ID of the query

        Returns:
            Empty DataFrame with correct columns based on evaluation expectations
        """
        # =====================================================================
        # 给每个 query 返回 *有正确列名* 的空 DataFrame — SemBench evaluator 在 failed query 上
        # 仍需要 schema-aligned 空表用于结果对齐
        # =====================================================================
        # ⚠ 注意! Q7 在这里 schema 是 ["scoreSentiment", "count"], 但 _execute_q7 的实际逻辑
        # 是处理 "all pairs opposite sentiment" 返回 [id, reviewId1, reviewId2] — schema 对不上!
        # 详 _execute_q7 注释 — 看起来是 q_id 映射错位 (Q7 / Q8 schema 在某个版本里被对调过)
        if query_id == 1:
            # Q1: Five clearly positive reviews - returns reviewId
            return pd.DataFrame(columns=["reviewId"])
        elif query_id == 2:
            # Q2: Five positive reviews for "taken_3" - returns reviewId
            return pd.DataFrame(columns=["reviewId"])
        elif query_id == 3:
            # Q3: Count of positive reviews for "taken_3" - returns count
            return pd.DataFrame(columns=["count"])
        elif query_id == 4:
            # Q4: Positivity ratio for "taken_3" - returns ratio/average
            return pd.DataFrame(columns=["positivity_ratio"])
        elif query_id == 5:
            # Q5: Review pairs with same sentiment - returns id, reviewId1, reviewId2
            return pd.DataFrame(columns=["id", "reviewId1", "reviewId2"])
        elif query_id == 6:
            # Q6: Review pairs with opposite sentiment - returns id, reviewId1, reviewId2
            return pd.DataFrame(columns=["id", "reviewId1", "reviewId2"])
        elif query_id == 7:
            # Q7: Sentiment counts - returns scoreSentiment, count
            # ⚠ schema 对不上 _execute_q7 的输出 (后者期望 [id, reviewId1, reviewId2])
            return pd.DataFrame(columns=["scoreSentiment", "count"])
        else:
            # ⚠ Q8 没 case! _execute_q8 失败时 fallback 到空 DataFrame() 无列名
            return pd.DataFrame()

    def _execute_q1(self) -> pd.DataFrame:
        """
        Execute Q1: "Five clearly positive reviews (any movie)"

        Returns:
            DataFrame with columns: reviewId
        """
        # =====================================================================
        # Q1: top-5 review extraction. Pattern (Q1/Q2 共用):
        #   1. get_query_text 读 NL query 文本
        #   2. execute_caesura_query 跑 4-Phase 流水线拿 DataFrame
        #   3. schema-adapter: 找 "reviewId" 列 / fallback 用第一列 / 空时返回空 schema
        # =====================================================================
        # 读 SemBench 标准路径下的 NL query (files/movie/query/natural_language/Q1.txt)
        query_text = self.get_query_text(1, "natural_language")
        # 跑 CAESURA 拿结果 DataFrame (列名是 LLM 自由发挥的)
        results = self.execute_caesura_query(query_text)

        # Extract only reviewId column for evaluation
        # 三层 fallback 适配 LLM 输出的"列名不可预知":
        # 优先级 1: 精确匹配 "reviewId" 列
        if not results.empty and "reviewId" in results.columns:
            return results[["reviewId"]].head(5)
        # 优先级 2: 用第一列假设它就是 reviewId (rename)
        elif not results.empty:
            # Fallback: use first column if reviewId not found
            first_col = results.columns[0]
            return (
                results[[first_col]]
                .rename(columns={first_col: "reviewId"})
                .head(5)
            )
        # 优先级 3: 结果空 → 返回 schema-aligned 空 df
        else:
            return self._get_empty_results_dataframe(1)

    def _execute_q2(self) -> pd.DataFrame:
        """
        Execute Q2: "Five positive reviews for movie taken_3"

        Returns:
            DataFrame with columns: reviewId
        """
        # Q2 与 Q1 模式相同 (top-5 reviewId), 区别只在 NL query 内容 ("for movie taken_3")
        query_text = self.get_query_text(2, "natural_language")
        results = self.execute_caesura_query(query_text)

        # Extract only reviewId column for evaluation
        # 三层 fallback 同 Q1
        if not results.empty and "reviewId" in results.columns:
            return results[["reviewId"]].head(5)
        elif not results.empty:
            # Fallback: use first column if reviewId not found
            first_col = results.columns[0]
            return (
                results[[first_col]]
                .rename(columns={first_col: "reviewId"})
                .head(5)
            )
        else:
            return self._get_empty_results_dataframe(2)

    def _execute_q3(self) -> pd.DataFrame:
        """
        Execute Q3: "Count of positive reviews for movie taken_3"

        Returns:
            DataFrame with columns: count
        """
        # =====================================================================
        # Q3: 单数字聚合 (count). Schema adapter:
        # 优先级 1: 列名含 "count" (e.g. "count", "num_reviews_count", "review_count")
        # 优先级 2: 任何 numeric 列 (e.g. LLM 写 "total" / "n" / "result")
        # 优先级 3: 返回 count=0 单行 (非空 df, 与 Q1/Q2 的空 schema 不同; evaluator 把 0 当成功响应)
        # =====================================================================
        query_text = self.get_query_text(3, "natural_language")
        results = self.execute_caesura_query(query_text)

        # For aggregation queries, extract the numeric result
        if not results.empty:
            # Look for count-related columns
            # 子串 "count" 大小写不敏感匹配 — 兼容 "Count" / "COUNT" / "review_count" 等
            count_cols = [
                col for col in results.columns if "count" in col.lower()
            ]
            if count_cols:
                return results[[count_cols[0]]].rename(
                    columns={count_cols[0]: "count"}
                )
            else:
                # Use first numeric column if available
                # select_dtypes 是 pandas 按 dtype 过滤列的 API; 这里取所有数值列
                numeric_cols = results.select_dtypes(include=["number"]).columns
                if len(numeric_cols) > 0:
                    return results[[numeric_cols[0]]].rename(
                        columns={numeric_cols[0]: "count"}
                    )

        # Return 0 count if no results
        # 注意! 与 Q1/Q2 不同 — 返回 {"count": [0]} 一行 (非空), evaluator 会把它当 "count=0" 处理
        return pd.DataFrame({"count": [0]})

    def _execute_q4(self) -> pd.DataFrame:
        """
        Execute Q4: "Positivity ratio (average of 0/1) for movie taken_3"

        Returns:
            DataFrame with columns: positivity_ratio
        """
        # =====================================================================
        # Q4: 单数字比例 (avg). Schema adapter:
        # 优先级 1: 列名含 "ratio" / "average" / "avg"
        # 优先级 2: 任何 numeric 列
        # 优先级 3: 0.0 单行 fallback
        # =====================================================================
        query_text = self.get_query_text(4, "natural_language")
        results = self.execute_caesura_query(query_text)

        # For ratio/average queries, extract the numeric result
        if not results.empty:
            # Look for ratio/average related columns
            # any(term in col for term in [...]): 子串多关键词匹配
            # 注意 "average" 包含 "avg" 子串 — 都匹配; LLM 写 "averages" 也匹配
            ratio_cols = [
                col
                for col in results.columns
                if any(
                    term in col.lower() for term in ["ratio", "average", "avg"]
                )
            ]
            if ratio_cols:
                return results[[ratio_cols[0]]].rename(
                    columns={ratio_cols[0]: "positivity_ratio"}
                )
            else:
                # Use first numeric column if available
                numeric_cols = results.select_dtypes(include=["number"]).columns
                if len(numeric_cols) > 0:
                    return results[[numeric_cols[0]]].rename(
                        columns={numeric_cols[0]: "positivity_ratio"}
                    )

        # Return 0.0 ratio if no results
        return pd.DataFrame({"positivity_ratio": [0.0]})

    def _execute_q5(self) -> pd.DataFrame:
        """
        Execute Q5: "Ten Pairs of reviews that express the *same* sentiment for movie with id 'ant_man_and_the_wasp_quantumania'"

        Returns:
            DataFrame with columns: id, reviewId1, reviewId2
        """
        # =====================================================================
        # Q5: self-join 配对 (10 行限制). Schema adapter:
        # 优先级 1: 列名含 "id" 但不含 "review" (movie id) + 列名含 "review" 且含 "id" (reviewId)
        # 优先级 2: 按位置取前 3 列
        # 返回 .head(10) — 限制 10 对
        # =====================================================================
        query_text = self.get_query_text(5, "natural_language")
        results = self.execute_caesura_query(query_text)

        # For pair queries, we need 3 columns: movie id, reviewId1, reviewId2
        if not results.empty and len(results.columns) >= 3:
            # Try to find appropriate columns by name
            # 条件 1: "id" 是子串且 "review" 不是 → movie id (e.g. "movie_id", "id", "movieId")
            id_cols = [
                col
                for col in results.columns
                if "id" in col.lower() and "review" not in col.lower()
            ]
            # 条件 2: "review" 是子串且 "id" 是子串 → reviewId 类型 (e.g. "reviewId1", "review_id_a")
            review_cols = [
                col
                for col in results.columns
                if "review" in col.lower() and "id" in col.lower()
            ]

            # 至少 1 个 movie id + 至少 2 个 reviewId 才走精确路径
            if len(id_cols) >= 1 and len(review_cols) >= 2:
                selected_cols = [id_cols[0], review_cols[0], review_cols[1]]
                return (
                    results[selected_cols]
                    .rename(
                        columns={
                            selected_cols[0]: "id",
                            selected_cols[1]: "reviewId1",
                            selected_cols[2]: "reviewId2",
                        }
                    )
                    .head(10)
                )
            else:
                # Fallback: use first 3 columns
                # 列名模糊匹配失败 → 按位置取前 3 列, 假设 LLM 输出顺序就是 (movie_id, review1, review2)
                cols = list(results.columns)[:3]
                return (
                    results[cols]
                    .rename(
                        columns={
                            cols[0]: "id",
                            cols[1]: "reviewId1",
                            cols[2]: "reviewId2",
                        }
                    )
                    .head(10)
                )

        return self._get_empty_results_dataframe(5)

    def _execute_q6(self) -> pd.DataFrame:
        """
        Execute Q6: "Pairs of reviews that express the *opposite* sentiment for movie with id 'ant_man_and_the_wasp_quantumania'"

        Returns:
            DataFrame with columns: id, reviewId1, reviewId2
        """
        # Q6 与 Q5 schema adapter 完全相同 (代码可以提到 helper, 这里 paper 复现选择 copy-paste);
        # 区别只在 NL query 内容 ("same sentiment" vs "opposite sentiment")
        query_text = self.get_query_text(6, "natural_language")
        results = self.execute_caesura_query(query_text)

        # Same logic as Q5 for pair queries
        if not results.empty and len(results.columns) >= 3:
            # Try to find appropriate columns by name
            id_cols = [
                col
                for col in results.columns
                if "id" in col.lower() and "review" not in col.lower()
            ]
            review_cols = [
                col
                for col in results.columns
                if "review" in col.lower() and "id" in col.lower()
            ]

            if len(id_cols) >= 1 and len(review_cols) >= 2:
                selected_cols = [id_cols[0], review_cols[0], review_cols[1]]
                return (
                    results[selected_cols]
                    .rename(
                        columns={
                            selected_cols[0]: "id",
                            selected_cols[1]: "reviewId1",
                            selected_cols[2]: "reviewId2",
                        }
                    )
                    .head(10)
                )
            else:
                # Fallback: use first 3 columns
                cols = list(results.columns)[:3]
                return (
                    results[cols]
                    .rename(
                        columns={
                            cols[0]: "id",
                            cols[1]: "reviewId1",
                            cols[2]: "reviewId2",
                        }
                    )
                    .head(10)
                )

        return self._get_empty_results_dataframe(6)

    def _execute_q7(self) -> pd.DataFrame:
        """
        Execute Q7: All Pairs of reviews that express the *opposite* sentiment for movie with id 'ant_man_and_the_wasp_quantumania'

        Returns:
            DataFrame with columns: id, reviewId1, reviewId2
        """
        # =====================================================================
        # Q7: 同 Q6 但 *无 .head(10) 限制* — 返回全部 pair (full join 结果)
        # Schema adapter 与 Q5/Q6 类似, 但匹配规则稍紧:
        #   id_cols: 列名精确等于 "id" 或 "movieid" (小写)
        #   review_cols: 子串 "reviewid" (注意! 不带下划线; LLM 写 "review_id" 不匹配)
        # =====================================================================
        # ⚠ schema 与 _get_empty_results_dataframe(7) 的 ["scoreSentiment", "count"] 不匹配 —
        # 详 _get_empty_results_dataframe 注释 "看起来 Q7 / Q8 schema 被对调过"
        query_text = self.get_query_text(7, "natural_language")
        results = self.execute_caesura_query(query_text)

        # For join queries, we need id and two reviewId columns
        if not results.empty:
            # Look for id and reviewId columns
            # 精确匹配 — col.lower() in {"id", "movieid"} (不是子串)
            # 比 Q5/Q6 更严; LLM 写 "movie_id" / "review_id" 这种就 fall through
            id_cols = [
                col
                for col in results.columns
                if col.lower() in ["id", "movieid"]
            ]
            # 子串匹配 — col.lower() 含 "reviewid" (无下划线版本)
            review_cols = [
                col for col in results.columns if "reviewid" in col.lower()
            ]

            if id_cols and len(review_cols) >= 2:
                return results[
                    [id_cols[0], review_cols[0], review_cols[1]]
                ].rename(
                    columns={
                        id_cols[0]: "id",
                        review_cols[0]: "reviewId1",
                        review_cols[1]: "reviewId2",
                    }
                )
            elif len(results.columns) >= 3:
                # Fallback: use first three columns
                cols = list(results.columns)[:3]
                return results[cols].rename(
                    columns={
                        cols[0]: "id",
                        cols[1]: "reviewId1",
                        cols[2]: "reviewId2",
                    }
                )

        return self._get_empty_results_dataframe(7)

    def _execute_q8(self) -> pd.DataFrame:
        """
        Execute Q8: "Calculate the number of positive and negative reviews for movie taken_3"

        Returns:
            DataFrame with columns: scoreSentiment, count
        """
        # =====================================================================
        # Q8: group-by count (sentiment → count). Schema adapter:
        #   优先级 1: 找一个 sentiment-related 列 + 一个 count 列
        #   优先级 2: 按位置取前 2 列
        # ⚠ _get_empty_results_dataframe(8) 没分支 → fallback 到空 DataFrame() 无列名
        # =====================================================================
        query_text = self.get_query_text(8, "natural_language")
        results = self.execute_caesura_query(query_text)

        # For group by queries, we need sentiment and count columns
        if not results.empty and len(results.columns) >= 2:
            # Look for sentiment and count columns
            # 子串多关键词匹配 sentiment 列 — LLM 写 "sentiment_label" / "score_class" 都可
            sentiment_cols = [
                col
                for col in results.columns
                if any(term in col.lower() for term in ["sentiment", "score"])
            ]
            count_cols = [
                col for col in results.columns if "count" in col.lower()
            ]

            if sentiment_cols and count_cols:
                return results[[sentiment_cols[0], count_cols[0]]].rename(
                    columns={
                        sentiment_cols[0]: "scoreSentiment",
                        count_cols[0]: "count",
                    }
                )
            elif len(results.columns) >= 2:
                # Fallback: use first two columns
                cols = list(results.columns)[:2]
                return results[cols].rename(
                    columns={cols[0]: "scoreSentiment", cols[1]: "count"}
                )

        return self._get_empty_results_dataframe(8)
