"""Evaluator for the MMQA dataset."""

# ============================================================================
# 教学注释 (Annotation Pass) — MMQA (MultiModal QA) scenario evaluator
# ============================================================================
"""
============================================================================
MMQA = paper 的 "MultiModal Question Answering" benchmark (image / table /
text 混合). 与 animals/cars 不同, ground truth 不是 SQL 跑出来的 dataframe,
而是预先准备的 JSON 文件 `query/natural_language/q<qid>.json`, 每个 JSON 含
一个 'ground_truth' 字段 (list 或 dict).

本文件做 3 件事:

1. **`_get_ground_truth(qid)`**: 直接 copy GT JSON 文件到 `raw_results/ground_truth/Q<qid>.json`,
   返回原 path (不读内容).

2. **`compute_metrics(results, ground_truth)`** (模块级函数): 计算 P/R/F1.
   - tp = 命中数, fp = 错命中数; precision = tp/(tp+fp), recall = tp/|gt|
   - 注意 tp+fp 必须 == len(results) (即 results 里每个都被分类成 TP 或 FP)

3. **`_evaluate_qN(sys, gt_path)`** (7 个 query): per-query 各有差异:
   - Q1, Q3, Q5, Q6 (单列 set): 抽 system_results 的某列 (director / title /
     actor / Airlines) 跟 gt set 比.
   - Q2, Q7 (image_id 组合): 把 BigQuery 的 'uri' / Palimpzest 的 'filename'
     列名都改成 'image_id' (跨 engine 对齐); 再拆 path / 解码 %2e → '.',
     组成 (ID, image_id) 或 (Airlines, image_id) tuple set.
   - Q4 (genre, movie 组合): system_results 每行 movies_in_genre 用 ',' split,
     生成 (genre, movie) tuple set.
   - Q5 (兼容两种列名): system_results 可能有 '_output' (一些 engine 输出)
     或 'actor' (LOTUS 输出), 都接受.

★ paper §X.Y 中 MMQA 的 q1-q7 各 query 的 natural-language 描述见
[../mmqa_scenario.py](../mmqa_scenario.py) (有更详细的 query intent 说明).
★ query_id 也可能是 string "3a" / "3b" (variant), 这里 int(...[:-1]) 强制
转回数字 — 同一 query 的 a/b 变体共享 evaluator.

注释说明: 本注释 pass 只增加 comment, 不改任何原始代码.
============================================================================
"""

import json
import shutil
from typing import Union

import pandas as pd

from evaluator.generic_evaluator import (
    GenericEvaluator,
    QueryMetricAggregation,
    QueryMetricRetrieval,
    QueryMetricRank,
)


# ---------------------------------------------------------------------------
# 通用 set-based P/R/F1 (MMQA 7 个 query 都用这个), 不放进 GenericEvaluator
# 因为它接受 list 而不是 DataFrame (MMQA 的特殊性: GT 是 JSON 不是 SQL 结果).
# ---------------------------------------------------------------------------
def compute_metrics(results: list, ground_truth: Union[set, list]):
    tp = 0
    fp = 0

    for item in results:
        if item in ground_truth:
            tp += 1
        else:
            fp += 1

    assert tp + fp == len(
        results
    ), "True Positives and False Positives do not match the results length."
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(ground_truth) if len(ground_truth) > 0 else 0.0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return QueryMetricRetrieval(precision, recall, f1_score)


class MMQAEvaluator(GenericEvaluator):
    def __init__(self, use_case: str, scale_factor: int) -> None:
        super().__init__(use_case, scale_factor)

        self.use_case = use_case

    def _load_domain_data(self) -> None:
        pass

    def _get_ground_truth(self, query_id: int) -> str:
        src = self._root / "query" / "natural_language" / f"q{query_id}.json"

        # Copy ground truth JSON to raw_results/ground_truth/
        gt_dir = self._results_path / "ground_truth"
        gt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, gt_dir / f"Q{query_id}.json")

        return src

    def _evaluate_single_query(
        self, query_id: int, system_results: pd.DataFrame, ground_truth: str
    ) -> "QueryMetricRetrieval | QueryMetricAggregation | QueryMetricRank":
        """Evaluate a single query based on its type."""
        try:
            query_id = int(query_id)
        except ValueError:
            query_id = int(query_id[:-1])  # "3a" -> "3"
        evaluate_fn = self._discover_evaluate_impl(query_id)
        return evaluate_fn(system_results, ground_truth)

    def _evaluate_q1(
        self, system_results: pd.DataFrame, ground_truth_filepath: str
    ) -> QueryMetricRetrieval:
        results = []
        for _, row in system_results.iterrows():
            results.append(row["director"].strip(' "').lower())

        with open(ground_truth_filepath, "r") as f:
            ground_truth = {
                g.strip().lower() for g in json.load(f).get("ground_truth")
            }

        return compute_metrics(results, ground_truth)

    def _evaluate_q2(
        self, system_results: pd.DataFrame, ground_truth_filepath: str
    ) -> QueryMetricRetrieval:
        if "uri" in system_results.columns:  # for BigQuery
            system_results.rename(columns={"uri": "image_id"}, inplace=True)
        if "filename" in system_results.columns:  # for Palimpzest
            system_results.rename(
                columns={"filename": "image_id"}, inplace=True
            )

        results = set()
        for _, row in system_results.iterrows():
            image_id = row["image_id"].split("/")[-1]
            image_id = image_id.replace("%2e", ".")

            if len(row) == 2:
                results.add((row["ID"], image_id))
            elif len(row) == 3:
                results.add(
                    (
                        row["ID"],
                        image_id,
                        str(row["color"]).strip().lower(),
                    )
                )
            else:
                raise ValueError(
                    f"Unexpected number of columns: {len(row)} in the results."
                )

        with open(ground_truth_filepath, "r") as f:
            ground_truth = json.load(f).get("ground_truth")
            ground_truth = set(tuple(g) for g in ground_truth)

        return compute_metrics(results, ground_truth)

    def _evaluate_q3(
        self, system_results: pd.DataFrame, ground_truth_filepath: str
    ) -> QueryMetricRetrieval:
        results = system_results["title"].tolist()

        with open(ground_truth_filepath, "r") as f:
            ground_truth = set(json.load(f).get("ground_truth"))

        return compute_metrics(results, ground_truth)

    def _evaluate_q4(
        self, system_results: pd.DataFrame, ground_truth_filepath: str
    ) -> QueryMetricRetrieval:
        results = []
        for _, row in system_results.iterrows():
            genre = row["genre"].strip().lower()

            for movie in row["movies_in_genre"].split(","):
                results.append((genre, movie.strip().lower()))

        with open(ground_truth_filepath, "r") as f:
            raw_ground_truth = json.load(f).get("ground_truth")

        ground_truth = set()
        for genre, movies in raw_ground_truth.items():
            for m in movies:
                ground_truth.add((genre.strip().lower(), m.strip().lower()))

        return compute_metrics(results, ground_truth)

    def _evaluate_q5(
        self, system_results: list, ground_truth_filepath: str
    ) -> QueryMetricRetrieval:
        results = []
        for _, row in system_results.iterrows():
            if "_output" in row:
                results.append(row["_output"].strip().lower())
            elif "actor" in row:
                results.append(row["actor"].strip().lower())
            else:
                raise ValueError(
                    "Expected either '_output' or 'actor' column in the results."  # noqa: E501
                )

        with open(ground_truth_filepath, "r") as f:
            ground_truth = set(json.load(f).get("ground_truth"))
            ground_truth = {g.strip().lower() for g in ground_truth}

        return compute_metrics(results, ground_truth)

    def _evaluate_q6(
        self, system_results: pd.DataFrame, ground_truth_filepath: str
    ) -> QueryMetricRetrieval:
        results = system_results["Airlines"].tolist()

        with open(ground_truth_filepath, "r") as f:
            ground_truth = set(json.load(f).get("ground_truth"))

        return compute_metrics(results, ground_truth)

    def _evaluate_q7(
        self, system_results: pd.DataFrame, ground_truth_filepath: str
    ) -> QueryMetricRetrieval:
        if "uri" in system_results.columns:  # for BigQuery
            system_results.rename(columns={"uri": "image_id"}, inplace=True)
        if "filename" in system_results.columns:  # for Palimpzest
            system_results.rename(
                columns={"filename": "image_id"}, inplace=True
            )

        results = set()
        for _, row in system_results.iterrows():
            image_id = row["image_id"].split("/")[-1]
            image_id = image_id.replace("%2e", ".")
            results.add((row["Airlines"], image_id))

        with open(ground_truth_filepath, "r") as f:
            ground_truth = json.load(f).get("ground_truth", [])
            ground_truth = set(tuple(g) for g in ground_truth)

        return compute_metrics(results, ground_truth)
