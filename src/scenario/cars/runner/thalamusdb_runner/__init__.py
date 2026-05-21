# =============================================================================
# 教学注释 pass — cars/thalamusdb_runner package marker
# =============================================================================
# Pass-through re-export. 让 `from scenario.cars.runner.thalamusdb_runner import
# ThalamusDBRunner` 跟 `from .thalamusdb_runner.thalamusdb_runner import ...` 等价.
# __all__ 限定 `from ... import *` 只暴露 ThalamusDBRunner 一个名字.
#
# 其它 5 个 scenario (movie/animals/mmqa/medical/ecomm) 的 thalamusdb_runner 目录
# 没这个 __init__.py — 那些 scenario 走 namespace package 模式. cars 是历史遗留.
# =============================================================================
from .thalamusdb_runner import ThalamusDBRunner

__all__ = ["ThalamusDBRunner"]
