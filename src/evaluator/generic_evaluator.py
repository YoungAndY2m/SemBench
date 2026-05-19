"""
Created on June 2, 2025

@author: Jiale Lao

Generic evaluator base class for all use cases, you should implement
"generate_ground_truth" and "evaluate" functions for each query.
The function signatures for each query qi should be:
- self._generate_qi_ground_truth() -> pd.DataFrame:
- self._evaluate_qi()
- self._evaluate_qi()

============================================================================
教学注释 pass (CLAUDE.md §5.5 全面注释)
============================================================================

本文件定义 SemBench *evaluator 层* 的根抽象, 负责把 system 的 raw output
(raw_results/{system}/Q{id}.csv) 跟 ground truth 比较, 算出 *quality
metric* (准确度).

Quality metric 三大类 (按 query 性质)
-------------------------------------
1. **Retrieval (检索)** — query 返回一组 item, 看返多少 *相关*:
     precision = TP / (TP + FP) = "返的里面对的比例"
     recall    = TP / (TP + FN) = "应该返的里面找到的比例"
     F1        = 2 * P * R / (P + R) = P R 调和平均, 平衡两者
     (TP true positive 对返; FP 假阳, 不该返却返了; FN 假阴, 漏返)
   适合: sem_filter / sem_join / 关键词搜索类 query.

2. **Aggregation (聚合)** — query 返回单个数 (count / sum / avg), 看跟真值差:
     absolute_error                = |predicted - actual|
     relative_error                = absolute_error / |actual|
     mean_absolute_percentage_error = relative_error * 100
   适合: COUNT / SUM / AVG / MAX / MIN 类 query.

3. **Ranking (排序)** — query 返回带 score 的 item, 看 *排序顺序* 跟真值
   是否一致:
     Spearman ρ = 把两序列的 *rank* (1, 2, 3, ...) 求 Pearson 相关; ∈
                  [-1, 1], 1 = 完全同序, -1 = 完全反序, 0 = 无关.
     Kendall τ  = 对所有 (i, j) pair, 看 sys 与 GT 在 i, j 的相对顺序
                  是否一致; ∈ [-1, 1]. 比 Spearman 对 outlier 更稳健.
   适合: TOP-N 排序 / rank-by-relevance 类 query.

4. **Clustering (聚类)** — 特殊场景, query 返回 (id, category) pair, 看
   *分组* 跟真值是否一致:
     Adjusted Rand Index (ARI) = 比较两个 partition 的相似度 (考虑随机
                                  baseline), ∈ [-1, 1], 1 完美.
     Omega Index               = 处理 *overlapping* community (一个 id
                                  能属多组), 走 cdlib NodeClustering.
   适合: 把 sem_group_by 输出当 cluster 评估.

5 个 metric dataclass (头部声明)
--------------------------------
  QueryMetricRetrieval                     precision / recall / f1
  QueryMetricAggregation                   relative / absolute / MAPE
  QueryMetricRank                          Spearman / Kendall τ
  SingleAccuracyScore                      (accuracy, metric_type) — 通用形
  SingleAccuracyScoreWithRetrievalDetails  在 SingleAccuracyScore 上多带
                                            (precision, recall, f1)

evaluator 在 pipeline 里的位置
------------------------------
  run.py → runner.run_all_queries() → 写 metrics/{sys}.json + Q{id}.csv
  run.py → evaluator.evaluate_system(sys, queries)
       → 读 raw_results/{sys}/Q{id}.csv (system output)
       → 调 _get_ground_truth(qid) (子类实现, 通常 read 预先 dump 的 GT csv)
       → 调 _evaluate_single_query(qid, sys_df, gt_df) (子类实现, 路由到
                                                       相应 _evaluate_qX 方法)
       → 写回 metrics/{sys}.json (用 dict.update() merge, 不覆盖 runner
         已写的 row_count / time / token / cost 等字段)

依赖外部 metric 包
------------------
- sklearn.metrics: f1_score / adjusted_rand_score
- scipy.stats:     spearmanr / kendalltau
- cdlib:           NodeClustering / evaluation.omega (community detection
                   library; 用来算 overlapping clustering metric)

引用: [LOG_STRUCTURE.md §4.3 GenericEvaluator](../../LOG_STRUCTURE.md) +
       [§5.1 Quality Metrics](../../LOG_STRUCTURE.md)
============================================================================
"""

from __future__ import annotations

import abc
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional, Sequence
from cdlib import NodeClustering, evaluation

import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score


# ============================================================================
# QueryMetricRetrieval — sem_filter / sem_join 类 query 的 P/R/F1 容器
# ============================================================================
# precision = TP/(TP+FP); recall = TP/(TP+FN); f1 = 2PR/(P+R).
# 默认值都 0.0, 让"全错"或"未跑"的 default 状态有合理语义.
@dataclass
class QueryMetricRetrieval:
    """Metrics for retrieval tasks (e.g., finding relevant items)."""

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


# ============================================================================
# QueryMetricAggregation — COUNT / SUM / AVG 类 query 的误差容器
# ============================================================================
# 三个 field 关系:
#   absolute_error  = |predicted - actual|              [单位: 原始数据]
#   relative_error  = absolute_error / |actual|         [无单位, ≥ 0]
#   mean_absolute_percentage_error (MAPE)
#                   = relative_error * 100              [百分比]
# 三者 *同源*, calculate_errors() 一次设三个字段以保持一致.
@dataclass
class QueryMetricAggregation:
    """Metrics for aggregation tasks (e.g., counting, summing)."""

    relative_error: float = 0.0
    absolute_error: float = 0.0
    mean_absolute_percentage_error: float = 0.0

    # ========================================================================
    # calculate_errors — 一次性算 absolute / relative / MAPE 三个 field
    # ========================================================================
    # 边界处理 (actual = 0):
    #   - predicted 也 0 → 三个 err 都是 0 (完美). 严格意义"0/0 未定义",
    #     这里约定 0/0 = 0 (合理 baseline).
    #   - predicted != 0 → relative_error = inf (表示 "actual 全错").
    # MAPE 严格定义 (per Wikipedia) 假设 actual != 0; 这里加 if 防 0/0
    # 炸出 ZeroDivisionError.
    def calculate_errors(self, predicted: float, actual: float) -> None:
        """Populate the error fields based on *predicted* vs *actual*."""
        self.absolute_error = float(abs(predicted - actual))
        if actual != 0:
            self.relative_error = float(self.absolute_error / abs(actual))
            self.mean_absolute_percentage_error = float(
                self.relative_error * 100
            )
        else:
            self.relative_error = float("inf") if predicted != 0 else 0.0
            self.mean_absolute_percentage_error = float(
                self.relative_error * 100
            )


# ============================================================================
# QueryMetricRank — ranking 类 query 的两种相关系数 (Spearman / Kendall τ)
# ============================================================================
# 两种 *rank correlation* 都 ∈ [-1, 1]:
#   spearman: 对两个序列各自算 rank, 再求 rank 间的 Pearson 相关. 计算
#             复杂度 O(n log n) (sort), 对 outlier 比 Pearson 稳健.
#   kendall:  对 (i, j) 所有 pair 数 concordant / discordant pair 数. O(n²)
#             (有 O(n log n) 优化变种). 比 Spearman 对 small sample 更稳定,
#             但计算更慢.
# SemBench 选择两个都算, 供后续分析挑.
@dataclass
class QueryMetricRank:
    """Metrics for ranking tasks (e.g., scoring and ranking items)."""

    spearman_correlation: float = 0.0
    kendall_tau: float = 0.0


# ============================================================================
# SingleAccuracyScore — 统一的 single-float metric 容器
# ============================================================================
# 为什么这个 wrapper 存在: 不同 query 类型用不同 metric (P/R/F1 vs MAPE vs
# Spearman vs ARI), 后续 plot.py / analysis.py 要做"all query × all
# system × accuracy" 表格时, 需要把 *任意* metric 强转成一个 float +
# string 标签. 这个 dataclass 就是 *normalization* 层 — 不管底层是 F1 还是
# ARI, 都用 (accuracy, metric_type) 表示.
@dataclass
class SingleAccuracyScore:
    """
    Accuracy score class that can be used for any type of metric that produces a single float value.

    Makes post-processing easier, e.g., plotting, because it's transparent to the type of metric.
    """

    accuracy: float
    metric_type: str


# ============================================================================
# SingleAccuracyScoreWithRetrievalDetails — 在 SingleAccuracyScore 上多带
#                                            (precision, recall, f1) 三字段
# ============================================================================
# Python dataclass 支持 *继承* — class B(A) 时 B 自动继承 A 的所有 field,
# 还能加新 field. 这里 (accuracy, metric_type) 来自父类,
# (precision, recall, f1_score) 是子类新加.
# 用法: query 主指标 = f1 时, accuracy=f1 + 三个 detail 同时填; 后续
# plotting 主用 accuracy, drill-down 看 precision/recall 拆解.
@dataclass
class SingleAccuracyScoreWithRetrievalDetails(SingleAccuracyScore):
    """
    Special case for retrieval tasks that can contain more details on the accuracy metric.
    Not suitable for Adjusted-Rand-Index or Omega-Index
    """

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


# ============================================================================
# GenericEvaluator — 全 evaluator 的抽象基类 (ABC)
# ============================================================================
# 子类 (MovieEvaluator / EcommEvaluator / ...) 必须实现 3 个 @abstractmethod:
#   _load_domain_data()          init 时调一次, 缓存 domain CSV 到 self.*
#   _get_ground_truth(qid)       返 ground truth DataFrame
#   _evaluate_single_query(...)  把 (sys_df, gt_df) 转成 metric dataclass
#
# 父类提供 7 个 *generic_* 评估 helper, 子类的 _evaluate_qN 可以调:
#   _generic_retrieval_evaluation
#   _generic_aggregation_evaluation
#   _evaluate_tuple_matching
#   _evaluate_unique_values
#   _generic_ranking_evaluation
# 7 个 *compute_* static helper (没加 @staticmethod 装饰但实际是):
#   compute_precision / compute_recall / compute_f1_score /
#   compute_f1_score_classify / compute_adjusted_rand_index /
#   compute_omega_index / compute_accuracy_score (顶层 dispatcher)
class GenericEvaluator(abc.ABC):
    """Abstract base class for benchmark evaluators."""

    # ========================================================================
    # 3 个 @abstractmethod (子类必须 override): _load_domain_data /
    # _get_ground_truth / _evaluate_single_query — 抽象层负责把 evaluation
    # 流程跟具体 scenario 解耦.
    # ========================================================================
    @abc.abstractmethod
    def _load_domain_data(self) -> None:
        """Load domain-specific CSVs (called once during *init*)."""
        pass

    @abc.abstractmethod
    def _get_ground_truth(self, query_id: int) -> pd.DataFrame:
        """Return ground-truth DataFrame for *query_id*."""
        pass

    @abc.abstractmethod
    def _evaluate_single_query(
        self,
        query_id: int,
        system_results: pd.DataFrame,
        ground_truth: pd.DataFrame,
    ) -> "QueryMetricRetrieval | QueryMetricAggregation | SingleAccuracyScore":
        """Compute quality metrics and return an appropriate dataclass."""
        pass

    # ========================================================================
    # __init__ — 设 path 字段 + 调子类 _load_domain_data (init-time data 缓存)
    # ========================================================================
    # 字段:
    #   self._root          = <repo>/files/{use_case}/
    #   self._results_path  = <repo>/files/{use_case}/raw_results/
    #   self._metrics_path  = <repo>/files/{use_case}/metrics/
    # 与 GenericRunner 的 path 一致, evaluator 是 *读* runner 写下来的文件.
    # _load_domain_data 在 __init__ 末尾调 — 让子类把 cars.csv / reviews.csv
    # 等 domain CSV 缓存到 self.* (后续 _get_ground_truth 用).
    # Generic workflow – inherited by concrete evaluators
    def __init__(self, use_case: str, scale_factor: int) -> None:
        self._root = Path(__file__).resolve().parents[2] / "files" / use_case
        self._results_path = self._root / "raw_results"
        self._metrics_path = self._root / "metrics"
        self.scale_factor = scale_factor

        self._load_domain_data()

    # ========================================================================
    # evaluate_system — ★ 主入口 (run.py 调它)
    # ========================================================================
    # 模板方法顶层. 等价于 GenericRunner.run_all_queries 的 evaluator 版本.
    #
    # 流程:
    #   1. queries=None → 调 _discover_queries_for_system 扫 raw_results/.
    #   2. for each qid:
    #        a. _load_system_results → 读 raw_results/{sys}/Q{qid}.csv
    #        b. _get_ground_truth(qid) (子类实现)
    #        c. _evaluate_single_query(qid, sys_df, gt_df) → metric dataclass
    #        d. dataclasses.asdict(result) 转 dict, 加 query_id, append.
    #        e. 任何 exception → 记 {"query_id": qid, "error": "..."}.
    #   3. ★ dict.update() merge 模式写回 metrics/{sys}.json:
    #        - 读现有 JSON (runner 之前写的, 含 row_count / time / token /
    #          cost / model_name 等).
    #        - 对每 row, store.setdefault(key, {}).update(row) —— 把
    #          *新* metric 字段 (precision / recall / f1 / ...) merge 进去,
    #          *不* 覆盖 runner 写的字段.
    #        - json.dump 写回. ensure_ascii=False 让中文 / unicode 直接落盘
    #          不转义 (€ 等).
    #
    # 错误处理: 单 query crash 写 {"query_id": qid, "error": ...} 不带具体
    # metric 字段; 主进程继续跑后面 query, fail-soft.
    def evaluate_system(
        self, system_name: str, queries: Optional[Sequence[int]] = None
    ) -> None:
        """Evaluate *system_name* and persist metrics to `<system>.csv`."""

        if queries is None:
            queries = sorted(self._discover_queries_for_system(system_name))

        new_rows: List[Dict[str, Any]] = []  # collected for CSV

        # Iterate over queries--------------------------------------------------
        for qid in queries:
            print(f"Evaluating Q{qid} ...")
            try:
                sys_df = self._load_system_results(system_name, qid)
                gt_df = self._get_ground_truth(qid)
                result = self._evaluate_single_query(qid, sys_df, gt_df)

                # Convert dataclass → dict → row--------------------------------
                row = {"query_id": qid, **dataclasses.asdict(result)}
                new_rows.append(row)
            except Exception as exc:
                print(f"  Q{qid}: ERROR - {exc}\n{traceback.format_exc()}")
                new_rows.append({"query_id": qid, "error": str(exc)})

        # Write to json ---------------------------------------------------
        out_f: Path = self._metrics_path / f"{system_name}.json"
        out_f.parent.mkdir(parents=True, exist_ok=True)

        if out_f.exists():
            try:
                with out_f.open("r", encoding="utf-8") as fh:
                    store: Dict[str, Dict[str, Any]] = json.load(fh)
            except json.JSONDecodeError:
                store = {}
        else:
            store = {}

        for row in new_rows:
            key = f"Q{row['query_id']}"  # e.g. "Q1"
            # Merge into any existing entry instead of replacing it outright
            store.setdefault(key, {}).update(row)

        with out_f.open("w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2, ensure_ascii=False)

        print(f"[{self.__class__.__name__}] Metrics saved → {out_f}")

    # ========================================================================
    # _load_system_results / _discover_queries_for_system / _discover_*_impl
    # ========================================================================
    # 4 个 工具方法:
    #   _load_system_results          读 raw_results/{sys}/Q{qid}.csv
    #                                 (空文件 → 返空 DataFrame, 不抛错)
    #   _discover_queries_for_system  扫 raw_results/{sys}/ 列出存在的 Q*.csv
    #                                 (用 stem[1:].isdigit() 过滤非 Q\d 文件)
    #   _discover_ground_truth_impl   反射找子类的 _generate_qN_ground_truth
    #   _discover_evaluate_impl       反射找子类的 _evaluate_qN
    # 后两个用 getattr 走 per-query 方法路由, 与 GenericRunner.
    # _discover_query_impl 同模式.
    def _load_system_results(
        self, system_name: str, query_id: int
    ) -> pd.DataFrame:
        csv_f = self._results_path / system_name / f"Q{query_id}.csv"
        if not csv_f.exists():
            raise FileNotFoundError(csv_f)

        try:
            df = pd.read_csv(csv_f)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return df

    def _discover_queries_for_system(self, system_name: str) -> List[int]:
        folder = self._results_path / system_name
        return [
            int(f.stem[1:])
            for f in folder.glob("Q*.csv")
            if f.stem[1:].isdigit()
        ]

    def _discover_ground_truth_impl(self, query_id) -> callable:
        method_name = f"_generate_q{query_id}_ground_truth"
        try:
            query_fn = getattr(self, method_name)
            if not callable(query_fn):
                raise TypeError(f"{method_name} exists but is not callable")
            return query_fn
        except AttributeError:
            raise NotImplementedError(
                f"Query {query_id} not implemented for {self.system_name}."
            )

    def _discover_evaluate_impl(self, query_id) -> callable:
        method_name = f"_evaluate_q{query_id}"
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
    # _generic_retrieval_evaluation — 通用 retrieval P/R/F1 计算
    # ========================================================================
    # 假设输入两个 DataFrame, 用 *row 级 ALL-columns equality* 匹配 (即两行
    # 在所有共有列上完全相等 → match). 每个 GT row 最多匹配一次 (matched_gt
    # set 防重复).
    #
    # 边界:
    #   - GT 空: sys 也空 → P=1 (vacuously true); sys 非空 → P=0.
    #   - sys 空: 默认 QueryMetricRetrieval (全 0) — 此时 R=0 也合理.
    #
    # 复杂度: O(|sys| × |GT|) — 两层 iterrows, 适合小结果集.
    # iterrows 性能差但代码直白. 若 query 返回 1K+ rows 后续可换 hash-join.
    def _generic_retrieval_evaluation(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRetrieval:
        """
        Generic evaluation for retrieval queries WITHOUT limit clauses.

        Compares system results with ground truth and calculates precision, recall, and F1.
        """

        if len(ground_truth) == 0:
            return QueryMetricRetrieval(
                precision=1.0 if len(system_results) == 0 else 0.0
            )
        if len(system_results) == 0:
            return QueryMetricRetrieval()

        matches = 0
        matched_gt = set()
        for _, srow in system_results.iterrows():
            for gt_idx, gt_row in ground_truth.iterrows():
                if gt_idx in matched_gt:
                    continue
                common = set(srow.index) & set(gt_row.index)
                if all(
                    srow[c] == gt_row[c]
                    for c in common
                    if pd.notna(srow[c]) and pd.notna(gt_row[c])
                ):
                    matches += 1
                    matched_gt.add(gt_idx)
                    break
        precision = matches / len(system_results)
        recall = matches / len(ground_truth)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return QueryMetricRetrieval(precision, recall, f1)

    # ========================================================================
    # _generic_aggregation_evaluation — COUNT/SUM/AVG 类 query 的标量比较
    # ========================================================================
    # 期望 sys / GT 各只有 1 行 1 列 (或多列, 但只看第一个 numeric).
    # 流程:
    #   1. shape check: 不是 1 行 → 直接判错 (rel_err=1.0, MAPE=100%).
    #   2. first_num(df) 找第一列能转 float 的值. 用 LLM 输出时常出现 str
    #      "42" 而不是 int 42, 这里 try float() 再判 numeric.
    #   3. 拿到 sys_val + gt_val → 调 m.calculate_errors().
    # 任何中间 None → fail-soft 判 100% MAPE.
    def _generic_aggregation_evaluation(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricAggregation:
        """
        Generic evaluation for aggregation queries.

        Assumes single value comparison. Handles string-to-numeric conversion.
        """

        m = QueryMetricAggregation()
        if len(system_results) != 1 or len(ground_truth) != 1:
            m.relative_error = 1.0
            m.absolute_error = float("inf")
            m.mean_absolute_percentage_error = 100.0
            return m

        def first_num(df):
            for c in df.columns:
                val = df[c].iloc[0]

                # Try to convert to numeric if it's a string
                if isinstance(val, str):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        continue

                # Check if it's numeric after potential conversion
                if pd.api.types.is_numeric_dtype(type(val)) or isinstance(
                    val, (int, float)
                ):
                    return val
            return None

        sys_val, gt_val = first_num(system_results), first_num(ground_truth)
        if sys_val is None or gt_val is None:
            (
                m.relative_error,
                m.absolute_error,
                m.mean_absolute_percentage_error,
            ) = (1.0, float("inf"), 100.0)
            return m

        m.calculate_errors(predicted=sys_val, actual=gt_val)
        return m

    # ========================================================================
    # _evaluate_tuple_matching — 用前 n 列的值 *排序后* 组 tuple 做集合匹配
    # ========================================================================
    # 适合的 query: 返回 pair / triple / quadruple, 且 *顺序无关* (e.g.
    # sem_join 找两个 movie 都涉及同一明星 — 返 (A, B) 与返 (B, A) 应判同).
    # 实现:
    #   1. 取前 n 列, apply 把每行变成 sorted(tuple(row[col]...)).
    #      - sorted 让 (A, B) 与 (B, A) 等价 (无序对).
    #      - 任何列 NaN → None → 跳过该 row.
    #   2. set-of-tuples 做集合交集 → TP/precision/recall/F1.
    # set-based 复杂度 O(|sys|+|GT|) 比 _generic_retrieval_evaluation 快.
    def _evaluate_tuple_matching(
        self,
        system_results: pd.DataFrame,
        ground_truth: pd.DataFrame,
        n_columns: int,
    ) -> QueryMetricRetrieval:
        """
        Generic function to evaluate queries that require matching tuples of n
        columns. Works for pairs (n=2), triples (n=3), quadruples (n=4), etc.
        """

        # Fast both-empty check
        if system_results.empty and ground_truth.empty:
            return QueryMetricRetrieval(1.0, 1.0, 1.0)

        # If one is empty, handle per convention
        if system_results.empty and not ground_truth.empty:
            return QueryMetricRetrieval()
        if ground_truth.empty and not system_results.empty:
            return QueryMetricRetrieval()

        # Check if we have enough columns
        if len(system_results.columns) < n_columns:
            return QueryMetricRetrieval()

        # Use first n columns regardless of their names
        sys_cols = list(system_results.columns[:n_columns])
        gt_cols = list(ground_truth.columns[:n_columns])

        def create_tuple(row, cols):
            values = [row[col] for col in cols]
            if all(pd.notna(v) for v in values):
                return tuple(sorted(values))
            return None

        # Create sets of tuples from both dataframes
        sys_tuples = {
            t
            for t in system_results.apply(
                lambda r: create_tuple(r, sys_cols), axis=1
            )
            if t is not None
        }
        gt_tuples = {
            t
            for t in ground_truth.apply(
                lambda r: create_tuple(r, gt_cols), axis=1
            )
            if t is not None
        }

        # Calculate metrics
        correct = sys_tuples & gt_tuples
        precision = len(correct) / len(sys_tuples) if sys_tuples else 0.0
        recall = len(correct) / len(gt_tuples) if gt_tuples else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        return QueryMetricRetrieval(precision, recall, f1)

    # ========================================================================
    # _evaluate_unique_values — 把单列的 unique 值 set 做 P/R/F1 比较
    # ========================================================================
    # 适合: sem_filter 返回的"符合条件的 owner_name" — 顺序 / 重复无关,
    # 只看集合相等性.
    # 实现:
    #   1. 取指定列 (默认第 0 列), .dropna() 去空, 转 set.
    #   2. set intersection 得 TP.
    #   3. P = TP / |sys_set|, R = TP / |GT_set|, F1 = 2PR/(P+R).
    # 边界:
    #   - 两边都空 → P=R=F1=1 (vacuously true).
    #   - 一边空 → 全 0.
    def _evaluate_unique_values(
        self,
        system_results: pd.DataFrame,
        ground_truth: pd.DataFrame,
        column_index: int = 0,
    ) -> QueryMetricRetrieval:
        """
        Generic function to evaluate queries that compare unique values from a
        specific column.

        Used for queries like Q5, Q7, Q8, Q9, Q10 that compare owner names or
        similar.
        """

        # Fast both-empty check
        if system_results.empty and ground_truth.empty:
            return QueryMetricRetrieval(1.0, 1.0, 1.0)

        # If one is empty, handle per convention
        if system_results.empty and not ground_truth.empty:
            return QueryMetricRetrieval()
        if ground_truth.empty and not system_results.empty:
            return QueryMetricRetrieval()

        # Check if we have enough columns
        if len(system_results.columns) < column_index:
            return QueryMetricRetrieval()

        # Use the column at the specified index (default: first column)
        sys_col = system_results.columns[column_index]
        gt_col = (
            ground_truth.columns[column_index]
            if len(ground_truth.columns) > column_index
            else ground_truth.columns[0]
        )

        # Get unique values from both sets
        system_values = set(system_results[sys_col].dropna())
        ground_truth_values = set(ground_truth[gt_col].dropna())

        # Calculate true positives
        tp = len(system_values & ground_truth_values)

        # Calculate metrics
        precision = (
            (tp / len(system_values))
            if system_values
            else (1.0 if not ground_truth_values else 0.0)
        )
        recall = (
            (tp / len(ground_truth_values))
            if ground_truth_values
            else (1.0 if not system_values else 0.0)
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        return QueryMetricRetrieval(precision, recall, f1_score)

    # ========================================================================
    # _generic_ranking_evaluation — id+score 两列, 算 Spearman / Kendall τ
    # ========================================================================
    # 假设两 DataFrame 都有 *第 0 列 = id, 第 1 列 = score* 的 schema.
    # 流程:
    #   1. *延迟* import scipy.stats (避免 module-load 时强依赖 scipy).
    #   2. 各自把 (id, score) 转成 dict {id: float(score)}; 任何 NaN /
    #      非 numeric 跳过.
    #   3. 取 common_ids = sys_ids ∩ gt_ids.
    #   4. common_ids < 2 → 直接返 0 (相关系数对 n<2 没意义).
    #   5. 按 common_ids 顺序构造对齐数组 sys_values / gt_values.
    #   6. spearmanr / kendalltau 算系数; NaN (e.g. 所有 score 相同 → 无 rank
    #      可比) → 转 0.
    # 注意: ranking metric *不* 考虑 sys 或 GT 有但对方没有的 id (drop), 因为
    # 相关系数本质上是 *paired* sample 统计.
    def _generic_ranking_evaluation(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRank:
        """
        Generic evaluation for ranking queries.

        Assumes first column is the id and second column is the score/rank.
        Calculates Spearman's rank correlation coefficient and Kendall's tau coefficient.
        """
        from scipy.stats import spearmanr, kendalltau

        if len(system_results) == 0 or len(ground_truth) == 0:
            return QueryMetricRank(spearman_correlation=0.0, kendall_tau=0.0)

        # Ensure we have at least 2 columns (id, score)
        if len(system_results.columns) < 2 or len(ground_truth.columns) < 2:
            return QueryMetricRank(spearman_correlation=0.0, kendall_tau=0.0)

        # Use first column as id, second as score
        sys_id_col = system_results.columns[0]
        sys_score_col = system_results.columns[1]
        gt_id_col = ground_truth.columns[0]
        gt_score_col = ground_truth.columns[1]

        # Create dictionaries for mapping id to score
        sys_scores = {}
        for _, row in system_results.iterrows():
            id_val = row[sys_id_col]
            score_val = row[sys_score_col]
            if pd.notna(id_val) and pd.notna(score_val):
                try:
                    sys_scores[id_val] = float(score_val)
                except (ValueError, TypeError):
                    continue

        gt_scores = {}
        for _, row in ground_truth.iterrows():
            id_val = row[gt_id_col]
            score_val = row[gt_score_col]
            if pd.notna(id_val) and pd.notna(score_val):
                try:
                    gt_scores[id_val] = float(score_val)
                except (ValueError, TypeError):
                    continue

        # Find common IDs
        common_ids = set(sys_scores.keys()) & set(gt_scores.keys())
        if len(common_ids) < 2:
            return QueryMetricRank(spearman_correlation=0.0, kendall_tau=0.0)

        # Create aligned arrays for correlation calculation
        sys_values = [sys_scores[id_val] for id_val in common_ids]
        gt_values = [gt_scores[id_val] for id_val in common_ids]

        # Calculate correlations
        spearman_corr = 0.0
        kendall_corr = 0.0

        try:
            spearman_result = spearmanr(sys_values, gt_values)
            spearman_corr = (
                spearman_result.correlation
                if not pd.isna(spearman_result.correlation)
                else 0.0
            )
        except Exception:
            spearman_corr = 0.0

        try:
            kendall_result = kendalltau(sys_values, gt_values)
            kendall_corr = (
                kendall_result.correlation
                if not pd.isna(kendall_result.correlation)
                else 0.0
            )
        except Exception:
            kendall_corr = 0.0

        return QueryMetricRank(
            spearman_correlation=spearman_corr, kendall_tau=kendall_corr
        )

    # ========================================================================
    # compute_precision / compute_recall / compute_f1_score —
    # *id-column based* P/R/F1; 7 个 compute_* 都缺 @staticmethod 但实际
    # 用法是 GenericEvaluator.compute_precision(gt, result, "id") (类调用,
    # 不传 self). 这跟 [LOG_STRUCTURE.md §4.3] 提到的"static-method 风格"
    # 一致, 上游应加 @staticmethod 装饰; 不修, 仅 surface.
    # ========================================================================
    # 与 _generic_retrieval_evaluation 的差别: 这里假设两个 DataFrame 都有
    # 一个 *unique id 列* (默认 "id"), 用 id 集合做交集; 不比较其它列.
    # 适合 ecomm scenario — TOML 声明 accuracy_metric = "f1-score" + 数据
    # 有标准 id 列.
    #
    # 边界 (GT 空): result 也空 → P=1 (vacuously); result 非空 → P=0.
    def compute_precision(
        ground_truth: pd.DataFrame,
        query_result: pd.DataFrame,
        id_column: str = "id",
    ):
        """
        Computes the precision of the query result towards the ground truth.
        Assumes that both dataframes have an "id" column the uniquely identifies
        a row. The precision is computed based on the IDs in the result and the
        ground truth.
        """
        ground_truth_ids = (
            set(ground_truth[id_column]) if not ground_truth.empty else set()
        )
        result_ids = (
            set(query_result[id_column])
            if query_result is not None and not query_result.empty
            else set()
        )
        if len(ground_truth_ids) == 0:
            # If ground truth is empty, precision is 1.0 if result is also
            # empty, else 0.0
            return 1.0 if len(result_ids) == 0 else 0.0
        predicted_positives = len(result_ids)
        if predicted_positives == 0:
            return 0.0
        true_positives = len(result_ids & ground_truth_ids)
        return true_positives / predicted_positives

    def compute_recall(
        ground_truth: pd.DataFrame,
        query_result: pd.DataFrame,
        id_column: str = "id",
    ):
        """
        Computes the recall of the query result towards the ground truth.
        Assumes that both dataframes have an "id" column the uniquely identifies
        a row. The recall is computed based on the IDs in the result and the
        ground truth.
        """
        ground_truth_ids = (
            set(ground_truth[id_column]) if not ground_truth.empty else set()
        )
        result_ids = (
            set(query_result[id_column])
            if query_result is not None and not query_result.empty
            else set()
        )
        if len(ground_truth_ids) == 0:
            # If ground truth is empty, recall is 1.0 if result is also empty,
            # else 0.0
            return 1.0 if len(result_ids) == 0 else 0.0
        true_positives = len(result_ids & ground_truth_ids)
        actual_positives = len(ground_truth_ids)
        return true_positives / actual_positives

    def compute_f1_score(
        ground_truth: pd.DataFrame,
        query_result: pd.DataFrame,
        id_column: str = "id",
    ):
        """
        Computes the F1 score of the query result towards the ground truth.
        Assumes that both dataframes have an "id" column the uniquely identifies
        a row. The F1 score is computed based on the IDs in the result and the
        ground truth.
        """
        precision = GenericEvaluator.compute_precision(
            ground_truth, query_result, id_column=id_column
        )
        recall = GenericEvaluator.compute_recall(
            ground_truth, query_result, id_column=id_column
        )
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    # ========================================================================
    # compute_f1_score_classify — 多类分类 query 的 macro-F1
    # ========================================================================
    # 适合 sem_classify 类 query (e.g. "把每行 product 标记为 electronics /
    # books / clothing"). 与 compute_f1_score 不同:
    #   - 这里不算 set intersection, 而是 *逐 row* 比较 result_column 值.
    #   - 用 sklearn.metrics.f1_score(average="macro") — 各 class 独立算 F1
    #     再平均, 处理类别不均衡比 'micro' 更公平.
    # 强约束: GT 与 result 行数必须一致 (raise; 注意: raise "..." 是不合
    # 法 syntax, 应该 raise ValueError("...") — 这是 *实际 bug*, 跑到这行
    # 会抛 TypeError("exceptions must derive from BaseException"). surface
    # 不修.
    def compute_f1_score_classify(
        ground_truth: pd.DataFrame,
        query_result: pd.DataFrame,
        result_column: str,
        id_column: str = "id",
    ):
        """
        Computes the F1 score of the query result towards the ground truth.
        Assumes that both dataframes have an "id" column the uniquely identifies
        a row. The F1 score is computed based on the IDs in the result and the
        ground truth.
        """
        if ground_truth.shape[0] != query_result.shape[0]:
            raise "Invalid results. Ground truth and query vectors should be of the same length."

        gt = ground_truth.sort_values(id_column)[result_column]
        query = query_result.sort_values(id_column)[result_column]

        return f1_score(gt, query, average="macro")

    # ========================================================================
    # compute_adjusted_rand_index — 聚类 query 的 ARI
    # ========================================================================
    # Adjusted Rand Index 度量两个 partition (分组) 的相似度. 关键性质:
    #   - 跟原始 Rand Index 不同, ARI 减掉了 "随机 partition 的预期相似度",
    #     所以即便是完全随机的 partition, ARI 期望值 ≈ 0 (而 raw RI 在
    #     随机 partition 上会偏正).
    #   - 范围 ∈ [-1, 1]: 1 = 完美 (两 partition 相同, modulo label
    #     permutation), 0 = 随机 baseline, < 0 = 比随机还差.
    #   - 不假设两 partition 的 label 名一致 — sklearn 内部按结构匹配.
    # 用法: sem_group_by query 把每行划分到一个 category, 评估其分组是否
    # 跟 GT 接近. 注意 hard-code 用 "id" / "category" 列名.
    #
    # 实现 docstring 里说 "group" 列, 实际代码用 "category" 列 — surface
    # 标记, 不修.
    def compute_adjusted_rand_index(
        ground_truth: pd.DataFrame, query_result: pd.DataFrame
    ):
        """
        Computes the Adjusted Rand Index (ARI) between the group assignments in
        the query result and the ground truth. Assumes both dataframes have "id"
        and "group" columns.
        """
        # Handle None inputs gracefully
        if query_result is None:
            return 0.0

        # Map IDs to group assignments for both ground truth and query result
        gt_groups = ground_truth.set_index("id")["category"]
        qr_groups = query_result.set_index("id")["category"]

        common_ids = set(gt_groups.index) & set(qr_groups.index)
        if not common_ids:
            return 0.0

        gt_labels = [gt_groups[id] for id in common_ids]
        qr_labels = [qr_groups[id] for id in common_ids]
        return adjusted_rand_score(gt_labels, qr_labels)

    # ========================================================================
    # compute_omega_index — overlapping clustering 度量, 走 cdlib
    # ========================================================================
    # Omega Index (Collins & Dent 1988) 是 ARI 的 *overlapping community*
    # 扩展: 允许一个 item 属于多个 cluster. Adjusted Rand 假设 *硬分组*
    # (一 item 一组), 当 sem_group_by 返回 multi-label 时 ARI 不适用.
    #
    # cdlib (Community Discovery library) 提供 NodeClustering 表示 +
    # evaluation.omega 评估. 这里 graph=None (Omega 不需要 graph 结构,
    # 只需要 community lists).
    #
    # 实现:
    #   1. fillna("category", -1) — 把没分组的 item 当独立组.
    #   2. groupby("category")["id"].apply(list) — 把 (category → [ids])
    #      转成 list of lists 形式, 给 NodeClustering.
    #   3. evaluation.omega(pred_nc, gt_nc).score 返 float.
    def compute_omega_index(
        ground_truth: pd.DataFrame, query_result: pd.DataFrame
    ):
        # Convert to cluster lists. Consider items without category as a new group.
        ground_truth["category"] = ground_truth["category"].fillna(-1)
        query_result["category"] = query_result["category"].fillna(-1)

        gt_clusters = (
            ground_truth.groupby("category")["id"].apply(list).tolist()
        )
        pred_clusters = (
            query_result.groupby("category")["id"].apply(list).tolist()
        )

        gt_nc = NodeClustering(communities=gt_clusters, graph=None)
        pred_nc = NodeClustering(communities=pred_clusters, graph=None)

        return evaluation.omega(pred_nc, gt_nc).score

    # ========================================================================
    # compute_accuracy_score — 顶层 dispatcher: metric_type 字符串 → metric 值
    # ========================================================================
    # 由 ecomm scenario 的 EcommScenario.get_accuracy_measure_for_query 提供
    # metric_type 字符串 (例 "f1-score" / "precision" / "recall" /
    # "adjusted-rand-index" / "omega-index"), 这里按字符串路由调对应函数,
    # 返回统一的 SingleAccuracyScore (普通 metric) 或
    # SingleAccuracyScoreWithRetrievalDetails (F1/P/R 时带 drill-down).
    #
    # ★ 设计意义: 让 plot.py / analysis.py 不用关心 "这个 query 用哪个
    # metric", 一律拿 score.accuracy + score.metric_type. 这是
    # SingleAccuracyScore wrapper 的真正使命.
    #
    # 限制:
    #   - 加新 metric_type 要在这里加 elif (硬编码 + raise on unknown).
    #   - 不支持 aggregation metric (relative_error 等) 走 SingleAccuracyScore
    #     wrapper; aggregation query 当前用 _generic_aggregation_evaluation
    #     单独路线, 不通过本 dispatcher.
    def compute_accuracy_score(
        accuracy_metric_type: str,
        ground_truth: pd.DataFrame,
        query_result: pd.DataFrame,
        id_column: str = "id",
    ) -> SingleAccuracyScore:
        # Compute additional helper metrics if we have f1-score, precision, or recall
        if accuracy_metric_type in ["f1-score", "precision", "recall"]:
            f1_score = GenericEvaluator.compute_f1_score(
                ground_truth, query_result, id_column=id_column
            )
            precision = GenericEvaluator.compute_precision(
                ground_truth, query_result, id_column=id_column
            )
            recall = GenericEvaluator.compute_recall(
                ground_truth, query_result, id_column=id_column
            )

        if accuracy_metric_type == "f1-score":
            return SingleAccuracyScoreWithRetrievalDetails(
                f1_score,
                metric_type="f1-score",
                precision=precision,
                recall=recall,
                f1_score=f1_score,
            )
        elif accuracy_metric_type == "precision":
            return SingleAccuracyScoreWithRetrievalDetails(
                precision,
                metric_type="precision",
                precision=precision,
                recall=recall,
                f1_score=f1_score,
            )
        elif accuracy_metric_type == "recall":
            return SingleAccuracyScoreWithRetrievalDetails(
                recall,
                metric_type="recall",
                precision=precision,
                recall=recall,
                f1_score=f1_score,
            )
        elif accuracy_metric_type == "adjusted-rand-index":
            return SingleAccuracyScore(
                accuracy=GenericEvaluator.compute_adjusted_rand_index(
                    ground_truth, query_result
                ),
                metric_type="adjusted-rand-index",
            )
        elif accuracy_metric_type == "omega-index":
            return SingleAccuracyScore(
                accuracy=GenericEvaluator.compute_omega_index(
                    ground_truth, query_result
                ),
                metric_type="omega-index",
            )
        else:
            raise ValueError(
                f"Unsupported accuracy metric type: {accuracy_metric_type}"
            )
