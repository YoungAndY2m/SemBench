"""
FlockMTL runner implementation.

============================================================================
SemBench L1 wrapper — medical/runner/flockmtl_runner/flockmtl_runner.py
============================================================================
教学注释 pass (L1 ADD) by Claude.

⚠ **空壳 runner** — FlockMTL L1 4 个 scenario 中 medical 是不完整状态:
  - movie:   ✓ 完整 (调 setup_data 加载数据)
  - medical: ⚠ 本文件 — *没调 setup_data*, 只设 conn (没数据 → SQL 跑不通)
  - mmqa:    ❌ 全 dead code
  - cars:    ❌ runner 不存在

----------------------------------------------------------------------------
未使用的 import (多余, 复制粘贴遗留)
----------------------------------------------------------------------------
- typing.Any/Dict/List, overrides.override, pandas, time, jinja2.Environment, GenericQueryMetric:
  全部 import 了但 *没 1 个被本文件用*. 类似 setup 行 22 sys.path.append, 都是死代码.

----------------------------------------------------------------------------
比 movie runner 缺什么
----------------------------------------------------------------------------
movie runner 内 setup.setup_data(...) 一行 → 加载 2 个 CSV;
本文件没这行 → conn 内空表 → 执行 Q1.sql 会 "Table not found".
推测: dev 时 *手动* 用 setup.run_query() 跑了 setup_db.sql (medical 有这个文件); 那个文件含 DDL.
raw_results 里只有 Q1+Q10 因为只这两个跑通了.
============================================================================
"""

from pathlib import Path
import sys

# ⚠ 以下 7 个 import 全是 dead — 文件内没用到
from typing import Any, Dict, List
from overrides import override
import pandas as pd
from pathlib import Path
import os
import time
from jinja2 import Environment

from runner.generic_runner import GenericQueryMetric
from scenario.medical.setup.flockmtl import FlockMTLMedicalSetup
from runner.generic_flockmtl_runner.generic_flockmtl_runner import (
    GenericFlockMTLRunner,
)

# sys.path hack 同样无效 (见 movie runner 注释 — import 已经 eager 解析了)
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# jinja_env 定义了但 *本文件没用* — 父类 GenericFlockMTLRunner 内有自己的 jinja_env
jinja_env = Environment(variable_start_string="<<", variable_end_string=">>")


class FlockMTLRunner(GenericFlockMTLRunner):
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gpt-4o-mini",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup=skip_setup,
        )

        # ⚠ 这里 *缺 setup.setup_data(...)*; 仅创建 Setup 并取 conn — conn 内表为空
        # 对比 [movie runner] — 那有完整 setup_data; 本文件 dev 时漏写, 是已知 incomplete 实现
        self.flockmtl_conn = FlockMTLMedicalSetup().get_connection()
