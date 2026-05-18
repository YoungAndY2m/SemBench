# =============================================================================
# SemBench L1 vendored 副本 — caesura/model.py
# =============================================================================
# 教学注释 pass (L1 MODIFY) by Claude.
# 这是 L0 [AllSQPE/CAESURA/caesura/model.py](../../../../../../AllSQPE/CAESURA/caesura/model.py) 的 vendored 副本,
# 含 3 处 patch (相对 L0 +28 行), 加 **token usage 追踪** 让 SemBench 能算实际 LLM cost。
#
# Patch 位置 (本文件):
#   1. line 46-50      : 3 个 total_*_tokens 类字段              — MyOpenAI 新增计数器
#   2. line 52-64      : reset_token_usage() / get_token_usage() — 暴露给 Caesura wrapper 调
#   3. line 99-107     : _generate 末尾累加 usage                — 在每次真实 OpenAI 调用后捕获 usage
#
# ⚠ patch 不防 cache hit:
#   langchain SQLiteCache 命中时 *不* 调底层 super()._generate, 所以 result.llm_output 可能不带
#   token_usage 字段; 这里用 hasattr + key check 保护 (默默 fall through 不累加, 没 warn).
#   后果: cache hit 的 query token cost 显示为 0 — 见 [LOG.md pitfall #6](../../../../../../AllSQPE/CAESURA/LOG.md)
#
# byte-identical 部分 (rate limit / token clip / MyClient monkey-patch) 详 L0 教学注释。
# =============================================================================
import datetime
from pathlib import Path
import time
import logging
from typing import Any
from openai import Completion
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate


logger = logging.getLogger(__name__)

MAX_RPM = {
    "gpt-3.5-turbo-0613": 3_500,
    "gpt-4-0613": 200
}

MAX_TPM = {
    "gpt-3.5-turbo-0613": 90_000,
    "gpt-4-0613": 40_000
}

MAX_NUM_TOKENS_SOFT = {
    "gpt-3.5-turbo-0613": 4_096 - 1024,
    "gpt-4-0613": 4_096
}

MAX_NUM_TOKENS_HARD = {
    "gpt-3.5-turbo-0613": 4_096 - 1024,
    "gpt-4-0613": 8_192 - 1024
}

MULTIPLIER = 0.5
REDUCE_MULTIPLIER = False


class MyOpenAI(ChatOpenAI):
    last_call = 0
    logging_dir: Path = None
    start_time = datetime.datetime.now()
    call_counter = 0
    max_num_tokens_soft: int = 0
    max_num_tokens_hard: int = 0
    max_rpm: int = 0
    max_tpm: int = 0
    

    # =========================================================================
    # === L1 PATCH (line 46-50) ===
    # 3 个 token usage 计数器, 作为 pydantic class-level 字段声明 (因 ChatOpenAI 是 BaseModel).
    # L0 没这 3 个字段; L1 在每次真实 LLM 调用后累加 (见 _generate patch).
    # 单位: OpenAI API 返回的 token 数 (tiktoken-counted).
    # =========================================================================
    # Token usage tracking
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    

    # =========================================================================
    # === L1 PATCH (line 52-64) — 2 个新方法 (reset / get) 暴露给 Caesura wrapper
    # =========================================================================
    # Caesura.run() 开头调 reset_token_usage(), 结束后 GenericCaesuraRunner 调 get_token_usage()
    # 拿到 dict 后算 cost (见 generic_caesura_runner.py:_update_token_usage_and_cost).
    # =========================================================================
    def reset_token_usage(self):
        """Reset token usage counters."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
    

    def get_token_usage(self):
        """Get current token usage."""
        # 返回 dict 与 OpenAI API 原生 token_usage 字段同 key
        return {
            'prompt_tokens': self.total_prompt_tokens,
            'completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_tokens
        }

    def _generate(self, prompts, *args, **kwargs):
        if not isinstance(self.client, MyClient):
            self.max_num_tokens_hard = MAX_NUM_TOKENS_HARD[self.model_name]
            self.max_num_tokens_soft = MAX_NUM_TOKENS_SOFT[self.model_name]
            self.max_rpm = int(MAX_RPM[self.model_name] * MULTIPLIER)
            self.max_tpm = int(MAX_TPM[self.model_name] * MULTIPLIER)

            self.client = MyClient(self.client, self)

        num_tokens = self.get_prompt_len(prompts)
        while num_tokens > self.max_num_tokens_soft and len(prompts) > 3:
            prompts = prompts[:2] + prompts[3:]
            num_tokens = self.get_prompt_len(prompts)

        while num_tokens > self.max_num_tokens_hard:
                prompts[0].content = prompts[0].content[100:]
                if prompts[0].content == "":
                    raise ValueError("Prompt too long. No more possibility to shorten it. Abort!")
                num_tokens = self.get_prompt_len(prompts)

        current_call = time.time()
        delta = current_call - self.last_call

        requests_delay = 60 / self.max_rpm
        tokens_delay = (60 * num_tokens) / self.max_tpm
        sleep_time = max(0.0, requests_delay - delta, tokens_delay - delta)
        print(sleep_time)
        time.sleep(sleep_time)

        logger.debug(f"Request: {prompts}")
        result = super()._generate(prompts, *args, **kwargs)
        logger.debug(f"Response: {result}")
        

        # =====================================================================
        # === L1 PATCH (line 99-107) — 截取 OpenAI API 返回的 token_usage 累加
        # =====================================================================
        # langchain ChatResult.llm_output 是 dict, OpenAI 返回时含 'token_usage' 子 dict.
        # 三重防御:
        #   1. hasattr(result, 'llm_output')   防 result 是别的 schema
        #   2. result.llm_output 非 None       防 cache miss 后 wrapper 没填
        #   3. 'token_usage' in result.llm_output  防 cache hit 时这字段不存在
        # ⚠ 命中前 2 个 fail 时 *默默跳过* 不累加, 没 logger.warn — 详 file docstring "patch 不防 cache hit"
        # =====================================================================
        # Track token usage from API response
        if hasattr(result, 'llm_output') and result.llm_output and 'token_usage' in result.llm_output:
            usage = result.llm_output['token_usage']
            self.total_prompt_tokens += usage.get('prompt_tokens', 0)
            self.total_completion_tokens += usage.get('completion_tokens', 0) 
            self.total_tokens += usage.get('total_tokens', 0)
            logger.debug(f"Token usage - Prompt: {usage.get('prompt_tokens', 0)}, "
                        f"Completion: {usage.get('completion_tokens', 0)}, "
                        f"Total: {usage.get('total_tokens', 0)}")

        if self.logging_dir is not None:
            time_dir = self.logging_dir / ".prompts" / self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
            time_dir.mkdir(parents=True, exist_ok=True)
            with open(time_dir / str(self.call_counter), "w") as f:
                print("\n\n--\n".join(f"{type(p).__name__}: {p.content}" for p in prompts), file=f)
                print("*" * 300, file=f)
                print(result.generations[0].text, file=f)
            self.call_counter += 1
            if self.call_counter > 50:
                raise Exception("Too many prompts generated. Failed.")


        self.last_call = current_call
        return result

    def get_prompt_len(self, prompts):
        return self.get_num_tokens(ChatPromptTemplate.from_messages(prompts).format()) + 100


class MyClient(Completion):
    def __init__(self, client, llm):
        self._client = client
        self._llm = llm

    def create(self, *args, **kwargs):
        try:
            result = self._client.create(*args, **kwargs)
        except Exception as e:
            if REDUCE_MULTIPLIER:
                self._llm.max_rpm //= 2  # TODO also add possibility to increase rate again
                self._llm.max_tpm //= 2
            raise e
        return result

    def __getattr__(self, _attr):
        return getattr(self._client, _attr)

    def __setattr__(self, _name: str, _value: Any) -> None:
        if _name in ("_client", "_llm"):
            super().__setattr__(_name, _value)
        else:
            return setattr(self._client, _name, _value)
