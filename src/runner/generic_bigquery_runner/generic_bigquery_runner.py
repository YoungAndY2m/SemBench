"""
Generic BigQuery system runner implementation.

@Jiale update:
(1) cost calculation, including audio token, reasoning token.
(2) parameters for BigQuery SQL, including model specification and
thinking_budgets

============================================================================
教学注释 (Annotation Pass) — SemBench L1 ADD: GenericBigQueryRunner
============================================================================

本文件是 SemBench 把 Google BigQuery AI.GENERATE_*() 语义函数封装成
"SemBench engine runner" 的 *基类*. 285 LoC, 提供:

1. **`__init__`**: 建 BigQuery client (用 env var GCLOUD_PROJECT 找项目),
   附带 thinking_budget 参数 (Gemini 模型的"思考 token"预算, 0 表示不思考).

2. **`execute_queries(query_ids)`**: 核心方法 — 拿一批 query id, 对每条:
   - 生成唯一 run_uuid (UUID v4), 每条 query 一个 `{run_uuid}-q{query_id}` 标签
   - 用 jinja2 模板把 query_text 里的 <<connection>>, <<query_id>>,
     <<other_params>>, <<thinking_budget>> 占位符替换成具体值
     - 注意: jinja2 默认是 `{{ var }}`, 这里换成 `<<` `>>` 是为了避免跟
       BigQuery SQL 里的 `{}` 冲突
   - 调 `self.bq_client.query(...)` 跑 SQL, 收 dataframe
   - 把结果包成 GenericQueryMetric (SemBench 给所有 engine 的统一 metric 接口:
     status / execution_time / results / token_usage / money_cost / error)

3. **成本回收 (paper §L1 独家功能, generic_bigquery_runner.py 的核心改造)**:
   BigQuery 的 AI.GENERATE_* 函数把 LLM 调用日志写到 `inference_logs.*`
   表里. 跑完 query 后 sleep 5 秒等日志 materialize, 然后用 AGG_SQL 把这条
   query (按 run_uuid 过滤) 的所有 LLM 调用统计起来:
   - prompt_audio_tokens (语音输入 token)
   - prompt_other_tokens (文本/图像/视频输入 token)
   - output_tokens (输出 token)
   - reasoning_tokens (思考 token)
   再用 MODEL_PRICES 字典 (5 个 Gemini 模型的 per-1M-token 价格) 算总美元成本.
   - 重试逻辑: 如果日志还没 materialize → max 3 retries × 5s wait

**SemBench 给所有 engine 的统一抽象** (跟 LOTUS / Palimpzest / ThalamusDB 等
对齐):
- `get_system_name() -> str`: engine 名字, e.g. "bigquery" / "lotus" / "palimpzest"
- `execute_queries(query_ids) -> Dict[int, GenericQueryMetric]`: 跑一批 query
- BigQuery 的特殊性: 成本/token 不是从函数返回值拿, 而是从 inference_logs.*
  表二次查询拿 — 这是 BigQuery 的 audit log 机制决定的

**5 个 Gemini 模型价格**:
- gemini-2.5-pro (\$1.25/M input, \$10/M output) — 最贵, 最强
- gemini-2.5-flash (\$0.30/M input, \$2.50/M output) — 中端默认 model_name
- gemini-2.5-flash-lite (\$0.10/M input, \$0.40/M output) — 最便宜
- gemini-2.0-flash (\$0.15/M input, \$0.60/M output) — 旧版
- gemini-2.0-flash-lite (没列价格, 但 SQL 里抓得到)
- ★ 音频 input 比文本 input 贵 ~3 倍 (e.g. flash 0.30 vs 1.00)

**与其它 SQPE engine 对比**:
- LOTUS: 用本地 LLM client (OpenAI / vLLM), 成本/token 直接从 response 拿
- Palimpzest: 用 ExecutionStats, 成本是 NaiveCostModel 算的
- BigQuery: 用 inference_logs.* 表二次查询, 成本是 per-model SQL 累加

注释说明: 本注释 pass 只增加 comment, 不改任何原始代码 (CLAUDE.md §5.5 §D 规则).
============================================================================
"""

import os
import time
import uuid
from overrides import override
from typing import Dict, List

from google.cloud import bigquery
from jinja2 import Environment

from runner.generic_runner import GenericRunner, GenericQueryMetric

# jinja2 Environment = 模板渲染引擎实例; jinja2 默认占位符是 `{{ var }}` `{% if %}`,
# 这里改成 `<< var >>` 是为了避免跟 SQL 字符串里的 `{` `}` 冲突 (例: BigQuery
# SQL 表达式有时含 STRUCT<> 或 JSON_QUERY 的花括号).
jinja_env = Environment(variable_start_string="<<", variable_end_string=">>")


# ============================================================================
# GenericBigQueryRunner — BigQuery engine 在 SemBench 的 wrapper
# ============================================================================
# 继承 GenericRunner (SemBench 给所有 engine 的统一基类). 各 scenario 的
# bigquery_runner.py (animals / cars / ecomm / medical / mmqa / movie) 再继承
# 本类并加 scenario-specific 配置.
#
# Pipeline:
#   query_text (SQL 模板) -> jinja render -> templated_query (具体 SQL)
#     -> bq_client.query() -> dataframe results
#     -> sleep 5s -> AGG_SQL (从 inference_logs.* 拿 token + cost)
#     -> GenericQueryMetric (统一 metric 接口)
# ============================================================================
class GenericBigQueryRunner(GenericRunner):
    """Runner for BigQuery."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
        thinking_budget: int = 0,
    ):
        """
        Initialize BigQuery runner.

        Args:
            use_case: The use case to run
            model_name: LLM model to use
            thinking_budget: Budget for thinking tokens (default: 0)
        """
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )
        # thinking_budget = Gemini 2.5 系列的"思考 token"预算 (>0 触发模型先 think
        # 再 answer); 0 表示纯 inference 不思考 (节省时间 + 钱). 会传到 jinja 模板
        # 里作为 SQL function 参数.
        self.thinking_budget = thinking_budget

        # Set up BigQuery client (assumes GOOGLE_APPLICATION_CREDENTIALS is set)
        # google.cloud.bigquery.Client() = SDK 提供的连接对象; 需要环境变量
        # GOOGLE_APPLICATION_CREDENTIALS 指向 service-account JSON, 以及
        # GCLOUD_PROJECT 指定 GCP 项目 id.
        self.bq_client = bigquery.Client(
            project=os.environ.get("GCLOUD_PROJECT")
        )

    @override
    def get_system_name(self) -> str:
        # 'bigquery' 是 SemBench 在 metrics / log 文件里给本 engine 的标签;
        # 跟 LOTUS 的 'lotus', Palimpzest 的 'palimpzest' 同样定位.
        return "bigquery"

    @override
    def execute_queries(
        self, query_ids: List[int]
    ) -> Dict[int, GenericQueryMetric]:
        # === 第一阶段: 收集 query_text + 初始化 metric ===
        # uuid.uuid4() = 随机生成的 128-bit UUID v4 (e.g. "a1b2c3d4-..."), 每次
        # 跑都不同; 后面用作 `{run_uuid}-q{query_id}` 标签写到 SQL 的 query_uuid
        # 字段, 是 *本次 run* 把 LLM 调用日志和具体 query 关联起来的唯一 id.
        run_uuid = uuid.uuid4()
        # resolve this inconsistency later
        # 两路获取 query SQL:
        #   - 老路径: self._discover_query_text(query_id) (从 files/.../query/ 找)
        #   - 新路径: scenario_handler.get_query_text(qid, sys) (走 scenario 抽象)
        # SemBench 在 refactor 中, 两种共存.
        query_texts = {
            query_id: (
                self._discover_query_text(query_id)
                if self.scenario_handler is None
                else self.scenario_handler.get_query_text(
                    query_id, self.get_system_name()
                )
            )
            for query_id in query_ids
        }
        # 初始化每条 query 的 metric 容器, status="pending" 等待后面 fill.
        # GenericQueryMetric = SemBench 的统一 metric dataclass, 给所有 engine 用.
        query_metrics = {
            query_id: GenericQueryMetric(query_id=query_id, status="pending")
            for query_id in query_ids
        }

        # === 第二阶段: 逐条 query 跑 BigQuery SQL ===
        for query_id, query_text in query_texts.items():
            try:
                # Replace variable names in the query text
                # jinja2 render: 把 SQL 模板里的 `<<connection>>` 替换成
                # 'us.connection' (BigQuery AI.GENERATE 函数需要的 LLM
                # connection), `<<query_id>>` 替换成 `{run_uuid}-q{query_id}`
                # (标签, 用来在 inference_logs 里 join 回来), `<<other_params>>`
                # 替换成 `, endpoint => 'gemini-...'` (告诉 AI.GENERATE 用哪个
                # model), `<<thinking_budget>>` 替换成具体 int.
                templated_query = jinja_env.from_string(query_text).render(
                    connection="us.connection",
                    query_id=f"{run_uuid}-q{query_id}",
                    other_params=f", endpoint => '{self.model_name}'",
                    thinking_budget=self.thinking_budget,
                )

                # 计时 + 提交 SQL + 拉结果到 pandas DataFrame.
                # query_job = AsyncQueryJob 对象; .result() 阻塞等完成;
                # .to_dataframe() 把行集 download 成 pandas df.
                start_time = time.time()
                query_job = self.bq_client.query(templated_query)
                df = query_job.result().to_dataframe()
                execution_time = time.time() - start_time

                # 把结果填进 metric 容器, 标 success.
                query_metrics[query_id].results = df
                query_metrics[query_id].execution_time = execution_time
                query_metrics[query_id].status = "success"
            except Exception as e:
                # 任何异常 (SQL 语法错 / 配额超 / 网络断 / ...) → 标 failed,
                # 后面 cost 阶段会 skip 这条.
                print(
                    f"  Error executing query {query_id}: {type(e).__name__}: {e}"  # noqa: E501
                )
                query_metrics[query_id].status = "failed"
                query_metrics[query_id].error = str(e)

        # === 第三阶段: 从 inference_logs.* 抓 token 用量 + 算成本 ===
        # 这是 BigQuery wrapper 独有的逻辑: AI.GENERATE() 函数的 token / 价格
        # 不会出现在 SELECT 返回值里, 而是写到 inference_logs.<model> 表里.
        # 必须跑完 query 后查 inference_logs 才能拿到 cost.
        #
        # Prices per 1M tokens (USD). Input is split into audio vs "other"
        # (text/image/video).
        # 来源: https://cloud.google.com/vertex-ai/generative-ai/pricing
        # (2025 年 8 月快照, 可能过时)
        # 注意: 'output' 价格不区分 audio / other, 也包含 reasoning ("thoughts")
        # token (后面公式 total_output + total_reasoning 一起乘 output 价).
        MODEL_PRICES = {
            "gemini_2_5_pro": {
                "input_other": 1.25 / 1e6,
                "input_audio": 1.25 / 1e6,
                "output": 10.0 / 1e6,
            },
            "gemini_2_5_flash": {
                "input_other": 0.30 / 1e6,
                "input_audio": 1.00 / 1e6,
                "output": 2.50 / 1e6,
            },
            "gemini_2_5_flash_lite": {
                "input_other": 0.10 / 1e6,
                "input_audio": 0.30 / 1e6,
                "output": 0.40 / 1e6,
            },
            "gemini_2_0_flash": {
                "input_other": 0.15 / 1e6,
                "input_audio": 1.00 / 1e6,
                "output": 0.60 / 1e6,
            },
        }

        # AGG_SQL = 聚合查询, 按 query_uuid 标签从 5 个 inference_logs.<model>
        # 表里把本次 query 的 LLM 调用统计加总. 返回 (model_key, prompt_audio_tokens,
        # prompt_other_tokens, output_tokens, reasoning_tokens, ...) 每个 model
        # 一行.
        #
        # 结构:
        # 1. all_inference_logs CTE: UNION ALL 5 个 model 的 log 表, 给每行加一列
        #    model_key 标记是哪个 model.
        # 2. enriched CTE: 用 JSON_VALUE / JSON_QUERY_ARRAY 抽出 usageMetadata
        #    (Gemini API 返回的 token 计数), 再用子查询区分 audio vs other.
        # 3. 最外层 SELECT: 按 model_key GROUP BY + SUM 聚合.
        #
        # @query_uuid = jinja 没替换的 BigQuery 参数占位符, 后面 ScalarQueryParameter
        # 传入实际 `{run_uuid}-q{query_id}` 值. (这跟 jinja 不冲突, jinja 模板用
        # `<<>>` 是为 query_text, AGG_SQL 是固定字符串, 这里用 BigQuery 原生 `@` 参数.)
        AGG_SQL = """
        -- see SQL above; identical text
        WITH all_inference_logs AS (
        SELECT *, 'gemini_2_0_flash'        AS model_key FROM inference_logs.gemini_2_0_flash_001
        UNION ALL
        SELECT *, 'gemini_2_0_flash_lite'   AS model_key FROM inference_logs.gemini_2_0_flash_lite_001
        UNION ALL
        SELECT *, 'gemini_2_5_flash'        AS model_key FROM inference_logs.gemini_2_5_flash
        UNION ALL
        SELECT *, 'gemini_2_5_flash_lite'   AS model_key FROM inference_logs.gemini_2_5_flash_lite_preview_06_17
        UNION ALL
        SELECT *, 'gemini_2_5_pro'          AS model_key FROM inference_logs.gemini_2_5_pro
        ),
        enriched AS (
        SELECT
            model_key,
            full_response,
            SAFE_CAST(JSON_VALUE(full_response, '$.usageMetadata.promptTokenCount') AS INT64) AS prompt_total,
            COALESCE(ARRAY_LENGTH(JSON_QUERY_ARRAY(full_response, '$.usageMetadata.promptTokensDetails')), 0) AS prompt_details_len,
            SAFE_CAST(JSON_VALUE(full_response, '$.usageMetadata.candidatesTokenCount') AS INT64) AS output_tokens,
            SAFE_CAST(JSON_VALUE(full_response, '$.usageMetadata.thoughtsTokenCount')  AS INT64) AS reasoning_tokens,
            SAFE_CAST(JSON_VALUE(full_response, '$.usageMetadata.billablePromptUsage.textCount') AS INT64)  AS billable_text_count,
            SAFE_CAST(JSON_VALUE(full_response, '$.usageMetadata.billablePromptUsage.audioDurationSeconds') AS FLOAT64) AS billable_audio_seconds,
            (
            SELECT SUM(SAFE_CAST(JSON_VALUE(d, '$.tokenCount') AS INT64))
            FROM UNNEST(JSON_QUERY_ARRAY(full_response, '$.usageMetadata.promptTokensDetails')) AS d
            WHERE JSON_VALUE(d, '$.modality') = 'AUDIO'
            ) AS prompt_audio_detail_sum,
            (
            SELECT SUM(SAFE_CAST(JSON_VALUE(d, '$.tokenCount') AS INT64))
            FROM UNNEST(JSON_QUERY_ARRAY(full_response, '$.usageMetadata.promptTokensDetails')) AS d
            WHERE JSON_VALUE(d, '$.modality') != 'AUDIO'
            ) AS prompt_other_detail_sum
        FROM all_inference_logs
        WHERE JSON_VALUE(full_request, '$.labels.query_uuid') = @query_uuid
        )
        SELECT
        model_key,
        SUM( IFNULL( IF(prompt_details_len > 0, prompt_audio_detail_sum, 0), 0) ) AS prompt_audio_tokens,
        SUM( IFNULL( IF(prompt_details_len > 0, prompt_other_detail_sum, prompt_total), 0) ) AS prompt_other_tokens,
        SUM( IFNULL(output_tokens,   0) ) AS output_tokens,
        SUM( IFNULL(reasoning_tokens,0) ) AS reasoning_tokens,
        SUM( IFNULL(billable_text_count,  0) ) AS billable_text_count,
        SUM( IFNULL(billable_audio_seconds,0.0) ) AS billable_audio_seconds
        FROM enriched
        GROUP BY model_key
        """  # noqa: E501

        # 第三阶段主循环: 对每条成功的 query 跑 AGG_SQL 拿 cost.
        for query_id, metrics in query_metrics.items():
            # 跳过失败的 query (没 cost 可算).
            if metrics.status == "failed":
                continue

            # ★ 关键 race condition 处理: inference_logs.* 表是 *异步* 写入的,
            # AI.GENERATE 函数返回后日志可能延迟几秒才出现. 先 sleep 5s 保险.
            # Wait a moment for inference logs to materialize in BigQuery
            print(f"  Waiting 5 seconds for inference logs to materialize for query {query_id}...")
            time.sleep(5)

            # 容错: 5s 后还没到 → 最多重试 3 次, 每次再 sleep 5s, 共最多 15s 等.
            # 如果 max_retries 都失败, 把 token / cost 默认置 0.
            max_retries = 3  # 3 * 5s = 15s additional wait if needed
            retry_count = 0
            cost_retrieved = False

            while retry_count < max_retries and not cost_retrieved:
                try:
                    # 提交 AGG_SQL, 传 @query_uuid 参数. ScalarQueryParameter 是
                    # BigQuery SDK 给 SQL 参数化的方式 (类似 prepared statement),
                    # 避免 SQL injection + 节省 query plan 编译.
                    job = self.bq_client.query(
                        AGG_SQL,
                        job_config=bigquery.QueryJobConfig(
                            query_parameters=[
                                bigquery.ScalarQueryParameter(
                                    "query_uuid",
                                    "STRING",
                                    f"{run_uuid}-q{query_id}",
                                )
                            ]
                        ),
                    )
                    df = job.result().to_dataframe()

                    # df 空 = inference_logs 还没写入 → 当 race 处理, 抛异常触发重试.
                    if df.empty:
                        raise ValueError("usage not materialized yet (no rows)")

                    # df 是 per-model 行集; SemBench 总 token 数是跨所有 model 加总,
                    # 所以这里对各列 SUM. .fillna(0) 是把 NULL 当 0 处理 (例: model X
                    # 没 audio token → prompt_audio_tokens 是 NULL).
                    # Totals across all models for this query
                    total_prompt_other = int(
                        df["prompt_other_tokens"].fillna(0).sum()
                    )
                    total_prompt_audio = int(
                        df["prompt_audio_tokens"].fillna(0).sum()
                    )
                    total_output = int(df["output_tokens"].fillna(0).sum())
                    total_reasoning = int(
                        df["reasoning_tokens"].fillna(0).sum()
                    )

                    print(
                        f"{total_prompt_other}, {total_prompt_audio}, {total_output}, {total_reasoning}"  # noqa: E501
                    )

                    # 总 token = input (audio + other) + output + reasoning;
                    # 这跟 Gemini API 返回的 usageMetadata.totalTokenCount 应一致
                    # (但 BigQuery 这种二次查询路径不直接拿 totalTokenCount, 自己算).
                    # Token usage should match usageMetadata.totalTokenCount
                    # when present:
                    total_token_usage = (
                        total_prompt_other
                        + total_prompt_audio
                        + total_output
                        + total_reasoning
                    )

                    # 钱的算法: 每行 (= 每个 model) 按 MODEL_PRICES 查价, 然后加总.
                    # df.itertuples(index=False) = 把 pandas df 转成命名元组迭代器
                    # (比 .iterrows() 快 ~10x, 因为不重建 pandas Series 对象).
                    # Money: per-model pricing
                    total_cost = 0.0
                    for row in df.itertuples(index=False):
                        model = row.model_key
                        prices = MODEL_PRICES.get(model)
                        if not prices:
                            # Unknown model: count tokens but skip billing (or
                            # set a fallback if you prefer)
                            print(
                                f"  Warning: No pricing configured for model '{model}'. Cost will exclude this model."  # noqa: E501
                            )
                            continue

                        # input 成本: 文本/图像/视频 input 走 input_other 价,
                        # 音频 input 走 input_audio 价 (通常贵 3-7 倍).
                        in_cost = (
                            int(row.prompt_other_tokens) * prices["input_other"]
                        ) + (
                            int(row.prompt_audio_tokens) * prices["input_audio"]
                        )
                        # output 成本: 普通 output token + reasoning ("思考") token
                        # 都按 output 价计费. (Gemini 2.5 系列 reasoning 跟 output
                        # 同价; 老 model 没 reasoning, reasoning_tokens=0.)
                        # Reasoning tokens billed at output rate
                        out_cost = (
                            int(row.output_tokens) + int(row.reasoning_tokens)
                        ) * prices["output"]

                        total_cost += in_cost + out_cost

                    # 成功填入 metric, 跳出 retry loop.
                    metrics.token_usage = int(total_token_usage)
                    metrics.money_cost = float(total_cost)
                    cost_retrieved = True

                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(
                            f"  Error getting cost for query {query_id} (attempt {retry_count}/{max_retries}): {type(e).__name__}: {e}, retrying in 5 seconds..."  # noqa: E501
                        )
                        time.sleep(5)
                    else:
                        print(
                            f"  Final error getting cost for query {query_id}: {type(e).__name__}: {e}"  # noqa: E501
                        )

            # 全部 3 次重试都失败: 兜底 token=0, cost=0 (避免下游分析崩溃).
            # ★ 注意: 这会让聚合统计低估真实成本 (但 SemBench 会在 log 里 print
            # 警告, 用户可以从 log 找出哪些 query 缺 cost).
            if not cost_retrieved:
                print(
                    f"  Could not retrieve cost data for query {query_id} after {max_retries} attempts, setting to 0"  # noqa: E501
                )
                metrics.token_usage = 0
                metrics.money_cost = 0.0

        # 返回完整 metric dict: {query_id: GenericQueryMetric(status, results,
        # execution_time, token_usage, money_cost, error)}, 上层 GenericRunner
        # 会把它写到 SemBench 的结果 CSV / JSON.
        return query_metrics
