"""
ThalamusDB system runner implementation.
"""

# =============================================================================
# 教学注释 pass — ThalamusDB × ecomm scenario (★ 最简 stub: 39 LoC, override 1 方法)
# =============================================================================
# E-commerce scenario: 44K rows 文本商品数据 (Q1-Q12 都是文本 NLfilter).
#
# ⚠ 这是 6 个 ThalamusDB scenario runner 里最简的 — 仅 39 行:
#   - __init__: 不传 db_path (None) → base class 自动用 default thalamusdb.duckdb
#     在 scenario_handler.get_data_dir() 下
#   - 不建任何表, 不读 CSV — 假设 ecomm setup (Palimpzest / LOTUS 用的 DuckDB)
#     已经把表准备好了 (跟 cars/medical scenario 不一样, 它们都自己建表)
#   - 唯一 override: _discover_query_impl(query_id) → 直接读 ecomm 的 SQL 文件
#     (files/ecomm/query/thalamusdb/Q<id>.sql)
#
# @override decorator (Python 3.12+ typing.override): 编译期检查 _discover_query_impl
# 真的 override 了 base class 方法; 防 typo. 老 Python 版本可能没这个 import.
#
# default model = gpt-4o-mini (text-only, 不需要 audio/image).
#
# ⚠ LOG_STRUCTURE.md §10.3 bug: "ecomm runner 39 LoC stub" — 比其它 scenario 简略
# 太多, 实际跑可能 broken (e.g. setup 没真正 invoke). PLAN P0 实验前需要验证.
# =============================================================================

import os
from pathlib import Path
import sys
from typing import override

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_thalamusdb_runner.generic_thalamusdb_runner import (
    GenericThalamusDBRunner,
)


class ThalamusDBRunner(GenericThalamusDBRunner):
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
            None,
            skip_setup=skip_setup,
        )

    @override
    def _discover_query_impl(self, query_id) -> callable:
        sql_text = self.scenario_handler.get_query_text(
            query_id, self.get_system_name()
        )
        return lambda _: self.execute_thalamusdb_query(sql_text)
