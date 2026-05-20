"""
Created on May 28, 2025

@author: Jiale Lao

LOTUS system runner implementation based on generic_runner for movie use case.
"""

# ============================================================
# 教学注释 (L1 wrapper pass):
# ------------------------------------------------------------
# 这是 LOTUS L0 (AllSQPE/LOTUS/) 接入 SemBench 评测框架的核心 wrapper.
# 与 SemBench/src/runner/generic_runner.py 形成"模板方法"模式:
#   - GenericRunner 抽象基类 : 定义 execute_query / 评测 metric / 数据加载
#   - GenericLotusRunner     : 实现 LOTUS-specific 逻辑 (model 配置 + token
#                              抽取 + query 自动发现)
#
# LOTUS 是 PyPI 包 (`pip install lotus-ai==1.1.3`), L1 完全不动 L0 任何
# .py — 通过 `import lotus` 在 SemBench process 内调用. 与 CAESURA L1 不同
# (CAESURA L1 vendor 后 patch 3 文件).
#
# 关键设计点:
#   1) Per-model LM 配置 (line 100-147) : 每个 model_name 手工分支配
#      reasoning_effort / rate_limit 等 quirk; LOTUS upstream 不管这些,
#      L1 wrapper 沉淀经验
#   2) 双 cost bookkeeping (line ~300-308) : 同时记 LOTUS 内置 cost
#      + L1 自算 cost (基于 PRICING table); 生产路径用 L1 自算的
#   3) Warmup retry (line 149-177) : 启动时 3 次 exp backoff 防 cold start 429
#   4) Query 自动发现 (line 318-343) : regex 找 _execute_q\d+$ 方法
#
# 教学读法: 配 [AllSQPE/LOTUS/LOG_STRUCTURE.md §10.2](../../../../AllSQPE/LOTUS/LOG_STRUCTURE.md)
# 一起看, §10.2 给了 6 件 SemBench 在 LOTUS 之上加的事 (Per-query method
# dispatch + token cost double-bookkeeping + Per-model LM config zoo +
# Cascade wiring + Modality-aware join wiring + Warmup retry).
# ============================================================
from overrides import override
import pandas as pd
import time
from typing import List
import lotus
from lotus.models import LM
import re
from PIL import ImageFile

from runner.generic_runner import GenericRunner, GenericQueryMetric

# Allow loading of truncated images (some source images may be incomplete)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# PRICING table  ←  L1 自维护 model cost 表
# ------------------------------------------------------------
# 与 LOTUS L0 走 LiteLLM 内置 completion_cost 不同, L1 自己维护这张
# PRICING dict. 原因 ([AllSQPE/LOTUS/LOG.md pitfall #7](../../../../AllSQPE/LOTUS/LOG.md)):
#   - LiteLLM PRICING 是社区维护, 偶尔 systematically 偏离实际
#   - paper §7 实验需要 reproducible cost 数字
#   - L1 仅覆盖 paper 实验用的几个 model, 不全 (例 Anthropic / DeepSeek 没列)
#
# 单位: USD per 1M tokens. 字段:
#   - text   : text input token 费率
#   - audio  : audio input token 费率 (multimodal model 才有, 如 gpt-4o-audio)
#   - output : completion (LLM 输出) token 费率
#
# 一致性约定: model 名 lowercase 后 startswith 匹配 (例 "gemini-2.5-flash-lite"
# 配 "gemini-2.5-flash" 也算 prefix match). _calculate_cost (line ~190+)
# 用这个匹配规则.
# ============================================================
# Pricing rules: (text_input, audio_input, output) per 1M tokens
PRICING = {
    "gpt-4o": {"text": 2.5, "audio": 2.5, "output": 10.0},
    "gpt-4o-mini": {"text": 0.15, "audio": 0.15, "output": 0.6},
    "gpt-4o-audio-preview": {
        "text": 2.5,
        "audio": 2.5,
        "output": 10.0,
    },
    "gpt-5": {"text": 1.25, "audio": 1.25, "output": 10.0},
    "gpt-5-mini": {"text": 0.25, "audio": 0.25, "output": 2.0},
    "gemini-2.0-flash": {
        "text": 0.15,
        "audio": 1.0,
        "output": 0.6,
    },
    "gemini-2.5-flash": {
        "text": 0.3,
        "audio": 1.0,
        "output": 2.5,
    },
    "gemini-2.5-flash-lite": {
        "text": 0.1,
        "audio": 0.3,
        "output": 0.4,
    },
    "gemini-2.5-pro": {
        "text": 1.25,
        "audio": 1.25,
        "output": 10.0,
    },
}


# ============================================================
# Class: GenericLotusRunner  ←  L1 wrapper 主类
# ------------------------------------------------------------
# 继承 SemBench/src/runner/generic_runner.py:GenericRunner.
# 实现 4 个抽象方法 (template method pattern):
#   - get_system_name()  → "lotus"
#   - execute_query(qid) → tracks time / token / cost, dispatch 到 _execute_q<i>
#   - _discover_queries() → regex 找 _execute_q\d+$ 方法返 list[int]
#   - _get_empty_results_dataframe(qid) → query 失败时返回空 df 占位
#
# Constructor 参数:
#   - policy="approximate" / "exact" : 默认 approximate; movie/ecomm 此时
#     真正启用 cascade (wire cascade_args + helper LM + RM + VS); 其它
#     scenario 仍用 exact (全 oracle LM)
#   - ranking="map" / "topk" : 决定某些 query 用 sem_map 还是 sem_topk
#     做 ranking (movie Q9 等场景敏感)
#   - max_tokens=8192 写死 (line ~91): 长上下文 LLM (Gemini 200K) 用不上
#     完整能力 → SemBench cost 测量 systematically 偏低
#     [LOTUS/LOG.md pitfall #15](../../../../AllSQPE/LOTUS/LOG.md)
# ============================================================
class GenericLotusRunner(GenericRunner):
    """GenericRunner for LOTUS system."""

    # thinking for gemini-2.5-pro can not be disabled, but always has issues
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
        policy="approximate",
        ranking="map",
    ):
        """
        Initialize LOTUS runner.

        Args:
            use_case: The use case to run
            model_name: LLM model to use
        """
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )
        self.policy = policy  # the policy can be "exact" or "approximate"
        self.ranking = ranking  # the ranking method could be "topk" or "map", for sem_topk and sem_map respectively

        # Initialize LOTUS

        # to avoid exceeding the reasoning_token of gemini models, default in LOTUS is 512
        self.max_tokens = 8192

        # Configure LM based on model_name
        self.lm = self._configure_lm()

        lotus.settings.configure(lm=self.lm)

        self._initialize_lotus_with_warmup()

    # ========================================================
    # _configure_lm: per-model LM config zoo
    # --------------------------------------------------------
    # LOTUS upstream LM(model, **kwargs) 接受统一接口; 但实际不同 model
    # 在 LiteLLM provider 下行为差异大. L1 在这里手工分支:
    #
    #   - gemini-2.5-pro     : rate_limit=2000 (避免 concurrent=20 触发 429);
    #                           reasoning_effort="low" (避免"no content due
    #                           to length"). 注释说 "always has issues"
    #                           [LOTUS/LOG.md pitfall #7]
    #   - gemini-2.5-flash   : reasoning_effort="disable" (Flash 推理快)
    #   - gemini-2.0-flash   : 无 reasoning_effort (旧版不支持参数)
    #   - gpt-5-mini         : reasoning_effort="minimal" (o-series 推理 model)
    #   - 其它               : 默认 config, warning
    #
    # 这是 LOTUS upstream 没沉淀的经验, L1 wrapper 把"每 model 该怎么配"
    # 固化在这里, 用户切 model 时不用手动管.
    # ========================================================
    def _configure_lm(self) -> LM:
        """
        Configure Language Model based on self.model_name.

        Returns:
            Configured LM instance
        """
        base_config = {
            "max_batch_size": self.concurrent_llm_worker,
            "max_tokens": self.max_tokens,
        }

        model_lower = self.model_name.lower()

        if "gemini-2.5-pro" in model_lower or "gemini_2_5_pro" in model_lower:
            # gemini-2.5-pro: reasoning_effort="low", solve the "no content due to length" issue
            # but does not work when concurrent_llm_worker is 20, try using rate_limit to control
            return LM(
                self.model_name,
                rate_limit=2000,
                max_tokens=self.max_tokens,
                reasoning_effort="low",
            )
        elif (
            "gemini-2.5-flash" in model_lower
            or "gemini_2_5_flash" in model_lower
        ):
            # gemini-2.5-flash: reasoning_effort="disable", works well
            return LM(
                self.model_name, **base_config, reasoning_effort="disable"
            )
        elif (
            "gemini-2.0-flash" in model_lower
            or "gemini_2_0_flash" in model_lower
        ):
            # gemini-2.0-flash: no reasoning_effort parameter needed
            return LM(self.model_name, **base_config)
        elif "gpt-5-mini" in model_lower or "gpt_5_mini" in model_lower:
            # gpt-5-mini: reasoning_effort="minimal"
            return LM(
                self.model_name, **base_config, reasoning_effort="minimal"
            )
        else:
            # Default configuration for unknown models (no reasoning_effort)
            print(
                f"Warning: Unknown model '{self.model_name}', using default configuration"
            )
            return LM(self.model_name, **base_config)

    # ========================================================
    # _initialize_lotus_with_warmup: cold-start retry
    # --------------------------------------------------------
    # 某些 provider (Gemini / OpenAI) 在 SDK 长时间 idle 后 first call 会
    # 触发 429 / connection error. 这里跑一次 "hi" 试探调用, exp backoff
    # retry 3 次. 失败也不 raise, 只 warning continue — 让真实 query 时
    # 再 raise (避免 init 时永久卡死).
    #
    # max_retries=3 + retry_delay 翻倍 (1s, 2s, 4s) 是经验值, paper 没特别
    # 调过.
    # ========================================================
    def _initialize_lotus_with_warmup(self):
        """Initialize LOTUS and perform connection warmup to avoid first-query errors."""

        # Configure LOTUS
        lotus.settings.configure(lm=self.lm)

        # Strategy 1: Perform a simple warmup query
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Make a simple test call to establish connection
                self.lm.__call__(["hi"], show_progress_bar=False)
                print(
                    f"LOTUS connection warmed up successfully on attempt {attempt + 1}"
                )
                break

            except Exception as e:
                print(f"Warmup attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Don't fail initialization, just log the warning
                    print(
                        "Warning: Connection warmup failed, but continuing..."
                    )

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost based on prompt and completion tokens.
        
        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            
        Returns:
            Total cost in USD
        """
        model_name = self.model_name.lower()
        
        # Find matching pricing configuration
        pricing_config = None
        for model_key in PRICING:
            if model_key.lower() in model_name:
                pricing_config = PRICING[model_key]
                break
        
        if pricing_config is None:
            print(f"Warning: No pricing found for model '{self.model_name}', cost calculation skipped")
            return 0.0
        
        # Calculate cost: prices are per 1M tokens
        input_cost = (prompt_tokens / 1_000_000) * pricing_config["text"]
        output_cost = (completion_tokens / 1_000_000) * pricing_config["output"]
        total_cost = input_cost + output_cost
        
        return total_cost

    @override
    def get_system_name(self) -> str:
        return "lotus"

    # ========================================================
    # execute_query: 主入口, GenericRunner template method 实现
    # --------------------------------------------------------
    # 流程:
    #   1) reset_stats() : 清掉上一个 query 的 token 累计
    #   2) _discover_query_impl(query_id) : 找 _execute_q<i> 方法
    #   3) 计时 + 跑实际 query 函数
    #   4) 抽 token usage + cost 填到 metric
    #   5) 失败时 logger + 空 df + status="failed"
    #   6) finally: reset_stats 防止下次 query 误用本次累计
    #
    # GenericQueryMetric 含 query_id / status / execution_time / results /
    # token_usage / money_cost / error 字段.
    # ========================================================
    def execute_query(self, query_id: int) -> GenericQueryMetric:
        """
        Execute a specific query using LOTUS and return metric with results.

        Args:
            query_id: ID of the query (e.g., 1 for Q1, 5 for Q5)

        Returns:
            QueryMetric object containing results DataFrame and metrics
        """
        # Create appropriate metric object
        metric = GenericQueryMetric(query_id=query_id, status="pending")

        # Reset token stats before each query
        try:
            lotus.settings.lm.reset_stats()
        except Exception as e:
            print(f"  Warning: Could not reset stats: {e}")

        try:
            query_fn = self._discover_query_impl(query_id)
            start_time = time.time()
            results = query_fn()
            execution_time = time.time() - start_time

            # Store results in metric
            metric.execution_time = execution_time
            metric.results = results
            metric.status = "success"

            # Get token usage and cost
            self._update_token_usage(metric)

        except Exception as e:
            # Handle failure
            metric.status = "failed"
            metric.error = str(e)
            metric.results = self._get_empty_results_dataframe(query_id)
            print(f"  Error in Q{query_id} execution: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Reset stats after storing
            try:
                lotus.settings.lm.reset_stats()
            except:
                pass

        return metric

    def _get_empty_results_dataframe(self, query_id: int) -> pd.DataFrame:
        """
        Get empty DataFrame with correct columns for a query.

        Args:
            query_id: ID of the query

        Returns:
            Empty DataFrame with correct columns
        """
        if query_id == 1:
            return pd.DataFrame(columns=["reviewId", "movieId", "reviewText"])
        elif query_id == 5:
            return pd.DataFrame(
                columns=[
                    "id",
                    "reviewId_left",
                    "reviewText_left",
                    "reviewId_right",
                    "reviewText_right",
                ]
            )
        else:
            return pd.DataFrame()

    # ========================================================
    # _update_token_usage: 双 cost 账本
    # --------------------------------------------------------
    # 同时记 LOTUS 自己 (LiteLLM) 算的 cost + L1 自算 cost:
    #   - LOTUS cost (lotus.settings.lm.stats.physical_usage.total_cost):
    #     LiteLLM 走自己 PRICING DB 算的, 实时同步社区维护
    #   - Calculated cost (self._calculate_cost(prompt, completion)):
    #     L1 用 PRICING dict (line ~24-54) 算的, 固定快照
    #
    # metric 写入 calculated_cost (L1 自算) 而非 LOTUS 内置 — 生产路径选
    # L1 自算 (paper §7 reproducibility 优先).
    # 打印两数字让 user 对比, debug 差异.
    #
    # virtual_usage 没用 (LOTUS 双账本机制 [LOTUS/LOG.md pitfall #9]).
    # ========================================================
    def _update_token_usage(self, metric: GenericQueryMetric):
        """Update metric with token usage and cost information."""
        try:
            # Get usage stats from LOTUS using the correct API
            prompt_tokens = lotus.settings.lm.stats.physical_usage.prompt_tokens
            completion_tokens = lotus.settings.lm.stats.physical_usage.completion_tokens
            total_tokens = lotus.settings.lm.stats.physical_usage.total_tokens
            total_cost = lotus.settings.lm.stats.physical_usage.total_cost
            
            # Calculate cost using our token consumption
            calculated_cost = self._calculate_cost(prompt_tokens, completion_tokens)

            # Print both tokens for comparison
            print(f"prompt token: {prompt_tokens}, completion token: {completion_tokens}")
            print(f"total tokens: {total_tokens}")
            
            # Print both costs for comparison
            print(f"  LOTUS cost: ${total_cost:.6f}")
            print(f"  Calculated cost based on token consumption: ${calculated_cost:.6f}")

            metric.token_usage = total_tokens
            metric.money_cost = calculated_cost

        except Exception as e:
            print(f"  Warning: Could not get token usage: {e}")
            metric.token_usage = 0
            metric.money_cost = 0.0

    # ========================================================
    # _discover_queries: regex 自动发现 _execute_q<i> 方法
    # --------------------------------------------------------
    # SemBench 约定: 每个 query 一个 method 名 "_execute_q<int>", 例
    # _execute_q1 / _execute_q10. _discover_queries 用 dir(self) 列出全部
    # 属性 + regex 匹配, 返排序后的 query_id list.
    #
    # 好处: 加新 query 时不用维护 list, 写个新 method 自动加进来.
    # ========================================================
    def _discover_queries(self) -> List[int]:
        """
        Discover available queries for LOTUS.

        Any method named ``_execute_q<i>`` (where <i> is an integer ≥1) is treated
        as an implemented query.  The function returns the list of those integer
        IDs in ascending order.

        Returns:
            List of available query IDs
        """
        # Return only implemented queries

        pattern = re.compile(r"_execute_q(\d+)$")
        query_ids: List[int] = []

        # `dir(self)` lists all attribute names visible on the instance;
        # we then pick out callables whose names match our pattern.
        for attr_name in dir(self):
            match = pattern.match(attr_name)
            if match:
                attr = getattr(self, attr_name, None)
                if callable(attr):
                    query_ids.append(int(match.group(1)))

        return sorted(query_ids)
