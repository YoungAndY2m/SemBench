"""
Created on May 28, 2025

@author: Jiale Lao

Main entry point for running benchmarks on different multi-modal data systems.

Supports two execution modes:
1. Isolated mode (default when .venvs/ exists): Each system runs in its own
   virtual environment via subprocess, avoiding dependency conflicts.
2. Direct mode (--no-isolation or when .venvs/ doesn't exist): All systems
   run in the current Python process (legacy behavior).

============================================================================
教学注释 pass (CLAUDE.md §5.5 全面注释)
============================================================================

SemBench 整体定位
------------------
SemBench (Semantic query Benchmark) 是 *跨多模态语义查询引擎* (SQPE,
semantic query processing engine) 的 benchmark, 类比 TPC-H 之于 SQL OLAP.
本 repo 里的 "system" = SQPE = 一个能处理 "语义算子" (sem_filter / sem_join /
sem_map 等, 即 *把 LLM 当一个 row-level 算子*) 的引擎. 当前接入的 7 个
system: lotus / palimpzest / bigquery / snowflake / thalamusdb / flockmtl /
caesura (Sema 闭源 binary, 单独处理).

本文件 (src/run.py) 是 *全 repo 的 CLI 入口 + dispatcher*. 大致流程:
  1. argparse 收集用户 flag: --systems / --use-cases / --queries / ...
  2. load_dotenv (从 .env 读 API key / DB credential)
  3. 检测 .venvs/ 是否存在 → 决定执行模式 (isolated vs direct)
  4. 对每个 (use_case, system) 组合, 动态 import runner class, 调
     runner.run_all_queries() 拿 metric dict
  5. 对每个 use_case 动态 import evaluator class, 调
     evaluator.evaluate_system() 算 P/R/F1/相对误差等质量指标
  6. 打印 BENCHMARK SUMMARY, os._exit(0) 强终止

为什么需要两种执行模式 (execution mode)
---------------------------------------
不同 SQPE 引擎依赖冲突剧烈 — LOTUS 用 PyTorch 2.x + sentence-transformers,
Palimpzest 用自家 demonstrate framework + DSPy, ThalamusDB 用 DuckDB 加自家
patch, FlockMTL 是 DuckDB C++ extension. 同一 venv (virtual environment,
"虚拟环境" — Python 隔离依赖的标准做法, 每个 venv 有独立 site-packages)
装齐会 import error. 解决方案: 给每个 system 单独建 venv
(.venvs/{system}/), 主进程 (sembench venv) spawn subprocess 跑各 system,
通过 stdout marker (字符串标记) __WORKER_RESULT__...__END_WORKER_RESULT__
把 JSON 序列化的 metric 回传.

  | Mode     | 触发条件                              | 怎么跑                 |
  |----------|---------------------------------------|------------------------|
  | isolated | .venvs/{system}/bin/python 存在       | spawn subprocess       |
  | direct   | --no-isolation 或没 .venvs/{system}/  | 当前 Python 直接 import|

主要 pipeline 函数 (按文件出现顺序)
-----------------------------------
  get_runner_class(system, use_case)
      动态 import: scenario/{uc}/runner/{sys}_runner/{sys}_runner.py
  get_evaluator(use_case)
      动态 import: scenario/{uc}/evaluation/evaluate.py
  parse_query_ids(args)
      接受 ['1','5'] 或 ['Q1','Q5'] 两种格式, 归一成 [1, 5]
  get_system_venv_python(system)
      找 .venvs/{system}/bin/python (POSIX 路径; Windows 不支持)
  run_system_isolated(...)
      subprocess.run + stdout marker 协议解析 JSON 结果
  run_benchmark(systems, use_cases, ...)
      主双层循环 (use_case × system), isolated 不可用时 fallback direct
  main()
      argparse + 顶层 orchestration, 结尾 os._exit(0) 强终止背景线程

与本 repo 其他文件的关系
------------------------
- [src/run_worker.py]      isolated mode 在 child venv 里跑的脚本; 主进程通过
                           __WORKER_RESULT__ marker 解 JSON.
- [src/runner/generic_runner.py]
                           全 system 的根基类 (GenericRunner) +
                           GenericQueryMetric dataclass.
- [src/scenario/{uc}/{uc}_scenario.py]
                           每个 use case 自定义的 setup / data 钩子.
- [src/evaluator/generic_evaluator.py]
                           算质量 metric 的根基类 (GenericEvaluator) +
                           P/R/F1/relative-error/rank-corr 实现.
- [src/plot.py] / [scripts/analysis.py]
                           跑完 benchmark 后画图 + 出 paper Table.

总览参考: [LOG_STRUCTURE.md §5.1 入口与编排](../LOG_STRUCTURE.md)
"""

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENVS_DIR = PROJECT_ROOT / ".venvs"


# ============================================================================
# get_runner_class — 动态 import 把 (system, use_case) 解析成 runner 类
# ============================================================================
# 为什么用动态 import?
#   静态写 `from scenario.movie.runner.lotus_runner.lotus_runner import
#   LotusRunner` 会在 import 时立刻执行该模块的 top-level 代码 (包括 import
#   torch / sentence_transformers ...) — 在主 sembench venv 里那些 dep 没装,
#   import 会直接 crash. 用 importlib.import_module 把 import 延迟到真正
#   要跑这个 system 时, 这样 (a) 主进程启动快, (b) 没装的 system 不影响
#   其它 system.
#
# Module path 约定:
#   scenario.{use_case}.runner.{system}_runner.{system}_runner
#                                              ↑ 双层目录 + 文件同名
#   e.g. lotus on movie → scenario.movie.runner.lotus_runner.lotus_runner
#
# 返回:
#   - 成功: Type[GenericRunner] (callable, RunnerClass(use_case=..., ...) 实例化)
#   - 失败: None (打印 error, 调用方 skip 这个 system)
#
# 加新 system 时改动点 ([LOG_STRUCTURE.md §7.1](../LOG_STRUCTURE.md))
#   1. 这里加 mapping (本 dict)
#   2. config/system/{sys}/*.json 配置文件
#   3. scenario/{uc}/runner/{sys}_runner/{sys}_runner.py 实现 {Sys}Runner 类
#   4. 各 scenario 的 setup_scenario() 加 elif system == "{sys}"
def get_runner_class(system: str, use_case: str):
    """Dynamically import and return the runner class for a given system."""

    # Define system to runner class name mapping
    runner_classes = {
        "lotus": "LotusRunner",
        "palimpzest": "PalimpzestRunner",
        "bigquery": "BigQueryRunner",
        "snowflake": "SnowflakeRunner",
        "thalamusdb": "ThalamusDBRunner",
        "flockmtl": "FlockMTLRunner",
        "caesura": "CaesuraRunner",
    }

    if system not in runner_classes:
        raise ValueError(f"Unknown system: {system}")

    try:
        # Construct the module path using the fixed format
        module_path = (
            f"scenario.{use_case}.runner.{system}_runner.{system}_runner"
        )
        class_name = runner_classes[system]

        # Dynamically import the module and get the class
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    except ImportError as e:
        print(f"Error importing runner for {system} from {module_path}: {e}")
        return None
    except AttributeError as e:
        print(
            f"Error: Class {class_name} not found in module {module_path}: {e}"
        )
        return None


# ============================================================================
# get_evaluator — 动态 import (use_case → evaluator 类) 用来算 quality metric
# ============================================================================
# Runner 跑完只产出 row count / latency / token usage / cost; 真正的"对不对"
# (quality) 由 evaluator 计算 — 比较 system output 与 ground truth, 算
# precision / recall / F1 / relative error / Spearman rank correlation 等.
# 不同 use_case 的 query 类型不同, evaluation 逻辑也不同, 所以 evaluator
# class 也是按 scenario 拆开的.
#
# Module path 约定:
#   scenario.{use_case}.evaluation.evaluate           # 文件路径
#   class {UseCase}Evaluator                          # 类名 (CamelCase)
#   e.g. ecomm → scenario.ecomm.evaluation.evaluate.EcommEvaluator
#
# 返回值与 get_runner_class 对称: 成功返 class, 失败返 None.
#
# 加新 use_case 时要同步改 evaluator_classes dict + 落地 evaluate.py
# (详 [LOG_STRUCTURE.md §7.2](../LOG_STRUCTURE.md)).
def get_evaluator(use_case: str):
    """
    Get the evaluator for a specific use case.
    Dynamically imports the evaluator based on the use case.
    """
    # Define use case to evaluator class name mapping
    evaluator_classes = {
        "movie": "MovieEvaluator",
        "detective": "DetectiveEvaluator",
        "medical": "MedicalEvaluator",
        "animals": "AnimalsEvaluator",
        "ecomm": "EcommEvaluator",
        "mmqa": "MMQAEvaluator",
        "cars": "CarsEvaluator",
        # Add more use cases here as needed
    }

    if use_case not in evaluator_classes:
        raise ValueError(f"Unknown use case: {use_case}")

    try:
        # Construct the module path using the fixed format
        module_path = f"scenario.{use_case}.evaluation.evaluate"
        class_name = evaluator_classes[use_case]

        # Dynamically import the module and get the class
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    except ImportError as e:
        print(
            f"Error importing evaluator for {use_case} from {module_path}: {e}"
        )
        return None
    except AttributeError as e:
        print(
            f"Error: Class {class_name} not found in module {module_path}: {e}"
        )
        return None


# ============================================================================
# parse_query_ids — CLI 用户输入的 query 标识 → 排序后的 int list
# ============================================================================
# 支持两种写法 (一律转 int 再 sorted):
#   --queries 1 5 10        → [1, 5, 10]
#   --queries Q1 Q5 Q10     → [1, 5, 10]      # 论文里常写 Q1, Q2, ...
# 不合法输入 (Q1abc 之类) print warning + skip.
#
# 注意: 实际实现里如果走 'else' 分支, arg 会以 str 形式被 append, 跟带 Q 前缀
# 的 int 形式混在一起. 调用方应只传纯数字 ['1','5'] 或纯 Q 前缀 ['Q1','Q5'],
# 不要混用 (代码没强校验, 但跟 [LOG_STRUCTURE.md §5.1] 的约定一致).
def parse_query_ids(query_args: List[str]) -> List[int]:
    """
    Parse query IDs from command line arguments.

    Args:
        query_args: List of query identifiers (e.g., ['1', '5'] or ['Q1', 'Q5'])

    Returns:
        List of query IDs as integers
    """
    query_ids = []
    for arg in query_args:
        # Handle both formats: '1' and 'Q1'
        if arg.startswith("Q"):
            try:
                query_id = int(arg[1:])
                query_ids.append(query_id)
            except ValueError:
                print(f"Warning: Invalid query format '{arg}', skipping")
        else:
            query_ids.append(arg)

    return sorted(query_ids)


# ============================================================================
# get_system_venv_python — 查找 isolated mode 下某 system 的 Python 解释器
# ============================================================================
# 路径约定: <repo_root>/.venvs/{system}/bin/python
#   - 路径硬编码 POSIX 形式 (bin/python), Windows 上要 Scripts\\python.exe,
#     SemBench 目前只支持 Linux / macOS.
#   - .venvs/ 由 scripts/setup_envs.sh 创建 (内部用 uv venv .venvs/{sys}
#     然后 uv pip install 各 system 的 requirements).
#
# 返回值: 存在 → Path 对象; 不存在 → None (调用方退回 direct mode).
def get_system_venv_python(system: str) -> Optional[Path]:
    """Return the Python executable path for a system's venv, or None."""
    venv_python = VENVS_DIR / system / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return None


# ============================================================================
# run_system_isolated — 在 child venv subprocess 里跑一个 system + 取 metric
# ============================================================================
# 整个 isolated mode 的核心. 流程:
#   1. 拼出 venv 的 python 路径 + worker script (src/run_worker.py).
#   2. 拼 argv: [venv_python, run_worker.py, --system X, --use-case Y, ...].
#   3. subprocess.run(...) 阻塞执行, capture stdout+stderr (合并到 stdout).
#   4. 扫 stdout 找两种 marker:
#         __WORKER_RESULT__<json>__END_WORKER_RESULT__   # 正常返回 dict
#         __WORKER_ERROR__<msg>__END_WORKER_ERROR__      # exception 信息
#      Worker 用 stdout marker 而不是 stdout/stderr 直接传 JSON, 是因为
#      child venv 里很多 library 会自己 print 日志到 stdout, 容易混淆;
#      用唯一字符串 token 包裹 payload 最简单可靠.
#   5. 如果两种 marker 都没有 + return code = 0 (worker 跑了但没写 marker),
#      fallback 读 files/{use_case}/metrics/{system}.json 这个磁盘落盘文件.
#
# 返回值:
#   - dict 形如 {"Q1": {...}, "Q2": {...}, ...}  正常结果
#   - dict 形如 {"error": "..."}                 异常 / 找不到结果
#
# 注意 subprocess 的 stdout/stderr=STDOUT 合并 → 单一文本流, 方便 grep marker.
# regex 用 .+? (non-greedy, "最短匹配") 避免一次 run 多个 marker 时贪心吞掉
# 多个 record (虽然目前一次 run 只 emit 一个 marker, 防御性写法).
def run_system_isolated(
    system: str,
    use_case: str,
    queries: Optional[List[int]],
    model_name: str,
    scale_factor: Optional[int],
    skip_setup: bool,
) -> Dict:
    """
    Run a system in its isolated virtual environment via subprocess.

    Returns:
        Dictionary of query results, or {"error": "..."} on failure.
    """
    venv_python = get_system_venv_python(system)
    if not venv_python:
        return {"error": f"No venv found for {system} at {VENVS_DIR / system}"}

    worker_script = PROJECT_ROOT / "src" / "run_worker.py"

    cmd = [
        str(venv_python),
        str(worker_script),
        "--system", system,
        "--use-case", use_case,
        "--model", model_name,
    ]
    if queries:
        cmd += ["--queries"] + [str(q) for q in queries]
    if scale_factor is not None:
        cmd += ["--scale-factor", str(scale_factor)]
    if skip_setup:
        cmd += ["--skip-setup"]

    print(f"  [isolated] Using venv: {venv_python}")

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Print worker output
    if result.stdout:
        # Filter out the worker result marker and print the rest
        for line in result.stdout.splitlines():
            if "__WORKER_RESULT__" in line or "__WORKER_ERROR__" in line:
                continue
            print(f"  [{system}] {line}")

    # Parse worker result from stdout
    if result.returncode == 0 and result.stdout:
        match = re.search(
            r"__WORKER_RESULT__(.+?)__END_WORKER_RESULT__", result.stdout
        )
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    # Parse error
    if result.stdout:
        match = re.search(
            r"__WORKER_ERROR__(.+?)__END_WORKER_ERROR__", result.stdout
        )
        if match:
            return {"error": match.group(1)}

    if result.returncode != 0:
        return {"error": f"Worker exited with code {result.returncode}"}

    # Fallback: try to read metrics from disk
    metrics_file = (
        PROJECT_ROOT / "files" / use_case / "metrics" / f"{system}.json"
    )
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)

    return {"error": "No results returned from worker"}


# ============================================================================
# run_benchmark — 主双层循环: 对 (use_case, system) 笛卡尔积跑 + evaluate
# ============================================================================
# 控制流 (伪码):
#   for use_case in use_cases:
#       print banner
#       for system in systems:
#           venv = look up .venvs/{system}/bin/python
#           if venv 存在 AND use_isolation:
#               result = run_system_isolated(...)        # subprocess
#           else:
#               # fallback: direct mode — 在主进程直接 import + 跑
#               runner_class = get_runner_class(...)
#               runner = runner_class(...)
#               metrics = runner.run_all_queries(queries=...)
#               result = {f"Q{qid}": m.to_dict() for qid, m in metrics.items()}
#           results[use_case][system] = result
#
#       # 跑完所有 system → 拿同样的 ground truth 算质量 metric
#       evaluator = get_evaluator(use_case)(use_case, scale_factor)
#       for system in systems:
#           evaluator.evaluate_system(system, queries=...)   # 落盘到磁盘
#
# 返回: 嵌套 dict {use_case: {system: {Q{id}: metric_dict, ...}}}.
#
# 关键决策点:
#   - fallback 不在意 .venvs/{sys} 不存在的情况, 也 *直接* 在主进程 import,
#     如果 dep 不全会抛 ImportError → 由内层 try/except 捕获并 mark error.
#   - evaluator 是 per-use_case 共享的; 在 use_case 内部对 systems 逐一调.
#     evaluator 通常 read 各 system 之前已写出的 metrics/{sys}.json, 算
#     quality 字段 (P/R/F1 等) 再 dict.update() 写回同一文件.
#   - 注意 use_isolation flag 与 venv_python 存在性是 *AND*: --no-isolation
#     立即关; 没有 .venvs/{sys}/ 则单独 fallback (并打印 hint).
def run_benchmark(
    systems: List[str],
    use_cases: List[str],
    queries: List[int] = None,
    skip_setup: bool = False,
    model_name: str = "gemini-2.5-flash",
    scale_factor: str = None,
    use_isolation: bool = True,
):
    """
    Run benchmarks for specified systems and use cases.

    Args:
        systems: List of system names to benchmark
        use_cases: List of use cases to run
        queries: Optional list of specific query IDs to run (e.g., [1, 5])
        skip_setup: Whether to skip setup phase
        model_name: Model name to use for systems that support it
        use_isolation: Use per-system venvs when available
    """
    results = {}

    for use_case in use_cases:
        print(f"\n{'='*60}")
        print(f"Running benchmarks for use case: {use_case}")
        print(f"{'='*60}")

        results[use_case] = {}

        # Run each system
        for system in systems:
            print(f"\n--- Running {system} ---")

            # Decide execution mode
            venv_python = get_system_venv_python(system) if use_isolation else None

            if venv_python:
                # Isolated execution via subprocess
                system_results = run_system_isolated(
                    system=system,
                    use_case=use_case,
                    queries=queries,
                    model_name=model_name,
                    scale_factor=scale_factor,
                    skip_setup=skip_setup,
                )
                results[use_case][system] = system_results

                if "error" not in system_results:
                    print(f"✓ {system} completed successfully (isolated)")
                else:
                    print(f"✗ Error running {system}: {system_results['error']}")

            else:
                # Direct execution in current process (legacy mode)
                if use_isolation:
                    print(
                        f"  No venv found for {system}, "
                        f"falling back to direct execution"
                    )

                runner_class = get_runner_class(system, use_case)
                if not runner_class:
                    print(f"Skipping {system} due to import error")
                    continue

                try:
                    runner = runner_class(
                        use_case=use_case,
                        scale_factor=scale_factor,
                        skip_setup=skip_setup,
                        model_name=model_name,
                    )
                    system_metrics = runner.run_all_queries(queries=queries)

                    results[use_case][system] = {
                        f"Q{query_id}": metric.to_dict()
                        for query_id, metric in system_metrics.items()
                    }

                    print(f"✓ {system} completed successfully")

                except Exception as e:
                    print(f"✗ Error running {system}: {e}")
                    import traceback

                    traceback.print_exc()
                    results[use_case][system] = {"error": str(e)}

        # Run evaluation
        print(f"\n--- Running evaluation for {use_case} ---")
        try:
            evaluator_class = get_evaluator(use_case)
            evaluator = evaluator_class(use_case, scale_factor)

            # Evaluate all systems
            for system in systems:
                if (
                    system in results[use_case]
                    and "error" not in results[use_case][system]
                ):
                    print(f"Evaluating {system}...")
                    evaluator.evaluate_system(system, queries=queries)

            print("✓ Evaluation completed successfully")

        except Exception as e:
            print(f"✗ Error during evaluation: {e}")
            import traceback

            traceback.print_exc()

    return results


# ============================================================================
# main — 顶层 orchestration: argparse → run_benchmark → 打印 summary → 强终止
# ============================================================================
# 步骤:
#   1. load_dotenv() — 从 ./.env 把 OPENAI_API_KEY / GEMINI_API_KEY /
#      GCP_PROJECT_ID / ... 注入 os.environ. 各 system runner 在内部从
#      os.environ 读取 (避免 hard-code key 到 repo).
#   2. argparse — 8 个 flag (--systems / --use-cases / --queries / ...).
#   3. parse_query_ids — 归一 '1' / 'Q1' 两种格式.
#   4. 检测 .venvs/ 自动决定 isolation; 用户可用 --no-isolation 强关.
#   5. print 配置 → run_benchmark → print summary table.
#   6. os._exit(0) 强终止 *所有线程*, 不跑 sys.exit(0) 因为后者只 raise
#      SystemExit; LOTUS 的 connection pool / sentence-transformer 的
#      tokenizer 会在背景线程持有 socket, 用 sys.exit 主线程退了但子线程
#      还活着 → process hang. os._exit 不调用 atexit handler, 不 flush
#      buffer (所以前面手动 sys.stdout.flush(); sys.stderr.flush()),
#      但能强杀所有线程, benchmark 这种 *跑完就结束* 的脚本可以接受.
#
# 输出格式 (summary 段):
#   USE_CASE_NAME:
#     system_name: ✅ Completed
#         Q1: ✅ 12.34s, 5 rows, 1234 tokens, $0.0123
#         Q2: ❌ 5.67s, Error: timeout
#     other_system: ❌ Failed - some error
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run benchmarks on multi-modal data systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all queries for LOTUS on movie use case
  python run.py --systems lotus --use-cases movie

  # Run specific queries (1 and 5) for multiple systems
  python run.py --systems lotus bigquery --queries 1 5

  # Run queries using Q-prefix notation
  python run.py --systems lotus --queries Q1 Q5 Q10

  # Force direct execution (skip venv isolation)
  python run.py --systems lotus --no-isolation
        """,
    )

    parser.add_argument(
        "--systems",
        nargs="+",
        default=["lotus"],
        help="Systems to benchmark (e.g., lotus bigquery)",
    )

    parser.add_argument(
        "--use-cases",
        nargs="+",
        default=["movie"],
        help="Use cases to run (e.g., movie amazon_product real_estate)",
    )

    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        help="Specific query IDs to run (e.g., 1 5 or Q1 Q5). If not specified, runs all queries.",  # noqa: E501
    )

    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip downloading and setting up the data sets for the specified use cases; used to speed up runs after the initial setup has been completed.",  # noqa: E501
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model name to use for systems that support it (default: gemini-2.5-flash)",
    )

    parser.add_argument(
        "--scale-factor",
        type=int,
        help="Factor to control the dataset size. Note that each use case has its own range for its respective scale factor.",  # noqa: E501
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="Disable per-system venv isolation (run all systems in current process)",
    )

    args = parser.parse_args()

    # Parse query IDs
    query_ids = None
    if args.queries:
        query_ids = parse_query_ids(args.queries)
        if not query_ids:
            print("Error: No valid query IDs provided")
            sys.exit(1)

    # Determine isolation mode
    use_isolation = not args.no_isolation
    if use_isolation and VENVS_DIR.exists():
        available_venvs = [
            d.name for d in VENVS_DIR.iterdir()
            if d.is_dir() and d.name != "sembench"
            and (d / "bin" / "python").exists()
        ]
        if available_venvs:
            print(f"Per-system venvs detected: {', '.join(available_venvs)}")
        else:
            use_isolation = False
    else:
        use_isolation = False

    print("Multi-Modal Data Systems Benchmark")
    print(f"Systems: {', '.join(args.systems)}")
    print(f"Use cases: {', '.join(args.use_cases)}")
    print(f"Model: {args.model}")
    print(f"Queries: {', '.join(map(str, query_ids)) if query_ids else 'All'}")
    print(f"Scale factor: {args.scale_factor}")
    print(f"Isolation: {'enabled' if use_isolation else 'disabled'}")

    # Run benchmark
    results = run_benchmark(
        systems=args.systems,
        use_cases=args.use_cases,
        queries=query_ids,
        skip_setup=args.skip_setup,
        model_name=args.model,
        scale_factor=args.scale_factor,
        use_isolation=use_isolation,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for use_case, systems_results in results.items():
        print(f"\n{use_case.upper()}:")
        for system, system_results in systems_results.items():
            if "error" in system_results:
                print(f"  {system}: ❌ Failed - {system_results['error']}")
            else:
                print(f"  {system}: ✅ Completed")
                if system_results:
                    # Sort by query ID for consistent output
                    sorted_results = sorted(
                        system_results.items(),
                        key=lambda x: (
                            x[0][1:] if x[0].startswith("Q") else x[0]
                        ),
                    )

                    for query_key, metrics in sorted_results:
                        if isinstance(metrics, dict):
                            query_id = metrics.get("query_id", query_key)
                            # Handle both formats: integer ID or "Q{id}" string
                            if isinstance(
                                query_id, str
                            ) and query_id.startswith("Q"):
                                display_id = query_id
                            else:
                                display_id = f"Q{query_id}"

                            status = metrics.get("status", "unknown")
                            time_str = (
                                f"{metrics.get('execution_time', 0):.2f}s"
                            )

                            if status == "success":
                                row_count = metrics.get("row_count", 0)
                                token_usage = metrics.get("token_usage", 0)
                                cost = metrics.get("money_cost", 0.0)

                                print(
                                    f"    {display_id}: ✅ {time_str}, {row_count} rows",  # noqa: E501
                                    end="",
                                )
                                if token_usage > 0:
                                    print(f", {token_usage} tokens", end="")
                                if cost > 0:
                                    print(f", ${cost:.4f}", end="")
                                print()
                            elif status == "failed":
                                error_msg = metrics.get(
                                    "error", "Unknown error"
                                )
                                print(
                                    f"    {display_id}: ❌ {time_str}, Error: {error_msg}"  # noqa: E501
                                )
                            else:
                                print(f"    {display_id}: {time_str}")

    # Flush output before force-terminating (os._exit skips buffer flush)
    sys.stdout.flush()
    sys.stderr.flush()

    # Force terminate all threads including background ones (LOTUS connection
    # pools)
    os._exit(0)


if __name__ == "__main__":
    main()
