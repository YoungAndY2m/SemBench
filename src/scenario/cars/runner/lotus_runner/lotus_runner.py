"""
Lotus system runner implementation for Cars scenario.
"""

# ============================================================
# 教学注释 (L1 wrapper pass):
# ------------------------------------------------------------
# Cars scenario 的 Code* mode wrapper. "Code* mode" 是 SemBench 设计的两
# 形态之一 ([LOTUS/LOG.md "关键发现 #3"](../../../../../../AllSQPE/LOTUS/LOG.md)):
#
#   - Code mode  : 每个 query 写在 LotusRunner 的 _execute_q<i> method 里
#                  (inline). movie / animals / mmqa 用这种.
#   - Code* mode : 每个 query 在独立文件 files/cars/query/lotus/Q{i}.py 里,
#                  runner 通过 importlib.util.spec_from_file_location 动态
#                  加载. cars / medical / ecomm 用这种.
#
# Code* mode 的优势:
#   - 每个 query 独立文件 → 学术 reviewer 更容易 review
#   - 用户可以单独修改某个 query 不动 runner
#
# 缺点:
#   - 必须为每个 Q<i> 写一个 _execute_q<i> stub method (line 38-66 重复 10 次)
#   - 动态 import 调用栈比直接 method 多 1-2 帧, 调试稍麻烦
#
# Cars 当前只有 5 个 Q file (Q1/Q3/Q4/Q8/Q10), 缺 Q2/Q5/Q6/Q7/Q9
# ([LOTUS/LOG.md 遗留问题 #4](../../../../../../AllSQPE/LOTUS/LOG.md)):
# 但 LotusRunner 仍 declare 全 10 个 method (line 38-66) — 调缺的 Q 会
# importlib 报错.
# ============================================================
from pathlib import Path
import sys
import importlib.util

# sys.path hack: 把 SemBench/src/ 加入 import 路径, 让 `from runner.generic_lotus_runner`
# 能 import 到. parents 链 = lotus_runner.py → lotus_runner → runner → cars → scenario
# (5 层). 这是 Python implicit package import 的常见 fix.
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_lotus_runner.generic_lotus_runner import GenericLotusRunner


class LotusRunner(GenericLotusRunner):
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        super().__init__(
            use_case, scale_factor, model_name, concurrent_llm_worker, skip_setup
        )

    # ========================================================
    # _load_and_execute_query: Code* mode 动态 import 核心
    # --------------------------------------------------------
    # 路径计算: Path(__file__).resolve().parents[5]
    #   = lotus_runner.py 的 5 层父目录 = SemBench 根目录
    #   parents[5] / "files" / "cars" / "query" / "lotus" / f"Q{i}.py"
    #
    # importlib.util:
    #   1) spec_from_file_location("query_1", "/path/to/Q1.py") → spec 对象
    #   2) module_from_spec(spec) → 空 module 对象
    #   3) spec.loader.exec_module(module) → 实际执行 Q1.py 加载所有 def
    #   4) query_module.run(...) 调 Q1.py 内的 run() 函数
    #
    # 注: 此处 spec.loader 在 Python 3.4+ 永远非 None (FileFinder), 但
    # type checker 可能 warn — 实际不需 None check.
    # ========================================================
    def _load_and_execute_query(self, query_id: int):
        """Dynamically load and execute a query from the query files."""
        query_file = Path(__file__).resolve().parents[5] / "files" / self.use_case / "query" / "lotus" / f"Q{query_id}.py"

        # Load the query module
        spec = importlib.util.spec_from_file_location(f"query_{query_id}", query_file)
        query_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(query_module)

        # Execute the query - pass files_path as data_dir
        return query_module.run(str(self.files_path), self.scale_factor)

    def _execute_q1(self):
        return self._load_and_execute_query(1)

    def _execute_q2(self):
        return self._load_and_execute_query(2)

    def _execute_q3(self):
        return self._load_and_execute_query(3)

    def _execute_q4(self):
        return self._load_and_execute_query(4)

    def _execute_q5(self):
        return self._load_and_execute_query(5)

    def _execute_q6(self):
        return self._load_and_execute_query(6)

    def _execute_q7(self):
        return self._load_and_execute_query(7)

    def _execute_q8(self):
        return self._load_and_execute_query(8)

    def _execute_q9(self):
        return self._load_and_execute_query(9)

    def _execute_q10(self):
        return self._load_and_execute_query(10)
