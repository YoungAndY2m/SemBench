"""
CAESURA runner for movie use case.

============================================================================
SemBench/src/scenario/movie/runner/caesura_runner/__init__.py
Package marker — re-export per-scenario CaesuraRunner 类
============================================================================

教学注释 pass (L1 ADD) by Claude.

把 CaesuraRunner re-export 到 package namespace, SemBench 主 dispatcher 可以:
    from scenario.movie.runner.caesura_runner import CaesuraRunner

`__all__` 是 Python 约定 — 控制 `from <pkg> import *` 时导出的名字白名单;
这里只导 `CaesuraRunner` 一个类, 隐藏其它内部 helper。
"""

from .caesura_runner import CaesuraRunner

__all__ = ['CaesuraRunner']