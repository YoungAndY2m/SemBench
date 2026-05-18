# =============================================================================
# SemBench/src/runner/generic_caesura_runner/__init__.py
# Package marker — 让 vendored `caesura/` 子目录能被 import
# =============================================================================
# 教学注释 pass (L1 ADD) by Claude.
#
# 目的: 把本目录加到 sys.path, 让 `caesura/` 子目录能作为顶层包 import 用:
#   from caesura.main import Caesura   ← 不必写 from generic_caesura_runner.caesura.main import ...
#
# ⚠ 注释里写 "so tdb can be imported" — copy-paste 自 generic_thalamusdb_runner/__init__.py 没改;
# 实际本目录管 caesura, 不是 tdb (ThalamusDB)。dead comment, 不影响功能。
#
# 这个 sys.path hack 在 generic_caesura_runner.py 里又做了一遍 (insert(0, _caesura_path)),
# 是 *双重保险* — 万一某个执行路径绕过 generic_caesura_runner.py 直接 import caesura.*, 这里也保证生效。
# =============================================================================
# Package initialization
import sys
import os

# Add this directory to sys.path so tdb can be imported as a top-level module
# ⚠ "tdb" 是 ThalamusDB 残留注释; 本目录实际管 caesura
_current_dir = os.path.dirname(__file__)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)