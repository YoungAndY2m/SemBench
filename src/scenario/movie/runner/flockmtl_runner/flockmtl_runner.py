"""
FlockMTL runner implementation.

============================================================================
SemBench L1 wrapper — movie/runner/flockmtl_runner/flockmtl_runner.py
============================================================================
教学注释 pass (L1 ADD) by Claude.

这个文件干什么 (一句话):
  FlockMTLRunner — movie scenario 专属 runner; 继承 GenericFlockMTLRunner, 在 __init__ 内
  调 FlockMTLMovieSetup 完成数据+模型 setup, 拿 conn 后填给 self.flockmtl_conn.

继承链:
  GenericRunner (SemBench 基类)
    ↓
  GenericFlockMTLRunner (generic_flockmtl_runner.py; 提供 execute_queries)
    ↓
  FlockMTLRunner (本文件; 仅提供 __init__ 完成 movie-specific setup)

4 个 scenario (movie/medical/mmqa/cars) 各有自己的子类, 但只有 movie 完整可用:
  - movie: ✓ 完整
  - medical: ⚠ runner 空壳 (没调 setup_data)
  - mmqa: ❌ import 路径错 + dead code
  - cars: ❌ setup 在但 runner 缺
============================================================================
"""

import os
from pathlib import Path
import sys

from scenario.movie.setup.flockmtl import FlockMTLMovieSetup
from runner.generic_flockmtl_runner.generic_flockmtl_runner import (
    GenericFlockMTLRunner,
)

# sys.path hack: parents[4] 跳到 src/, 让 from runner.* / from scenario.* 能 import
# 但 *line 14 在 import 之后* — sys.path 修改对已经 import 的模块无用; 应该放 line 4
# (Python from-import 是 eager 的, sys.path 改完了也不会重新解析 import)
# 实际上能跑是因为 SemBench 调 runner 时已经把 src/ 加进 PYTHONPATH 了 → 本 sys.path 行是死代码
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


class FlockMTLRunner(GenericFlockMTLRunner):
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gpt-4o-mini",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        # 父类 init 设字段 (use_case / model_name / ...); self.flockmtl_conn = None 占位
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup=skip_setup,
        )
        # 关键 3 步: 创建 Setup → 加载数据 → 把 conn 赋给父类期望字段
        setup = FlockMTLMovieSetup(model_name=model_name)
        setup.setup_data(
            data_dir=Path(__file__).resolve().parents[5] / "files" / "movie"
        )
        # 填上父类 self.flockmtl_conn (父类 execute_queries 用)
        self.flockmtl_conn = setup.get_connection()
