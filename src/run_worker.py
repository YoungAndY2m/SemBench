"""
Subprocess worker for running a single system in an isolated environment.

This script is invoked by run.py when per-system virtual environments are
detected. It runs a single system's queries and saves results to disk.
The main run.py process then reads those results for evaluation.

Usage (called automatically by run.py):
    .venvs/lotus/bin/python src/run_worker.py \
        --system lotus --use-case movie --queries 1 3 \
        --model gemini-2.5-flash --scale-factor 2000

============================================================================
教学注释 pass (CLAUDE.md §5.5 全面注释)
============================================================================

本文件在 SemBench pipeline 里的位置
------------------------------------
src/run.py (主进程, sembench venv)
    └── run_system_isolated(system, ...)
            └── subprocess.run([.venvs/{system}/bin/python, run_worker.py, ...])
                    └── ★ 本文件 ★
                            └── from run import get_runner_class
                                # 子 venv 里也能 import 主 src/run.py 因为
                                # 第 20 行把 src/ 加进了 sys.path

为什么需要这个独立 script (而不是 inline 在 run.py 里):
  - subprocess 需要一个 *可执行 script* 作为 argv[0]; importlib 在 subprocess
    边界没用 (subprocess 启动的是 *新 Python 进程*, 没有共享内存).
  - 子进程跑在 *某个 system 自家 venv* 里, 那个 venv 装了 LOTUS / Palimpzest /
    ThalamusDB 的具体 dep, 而主 sembench venv 没装. 这里 from run import 拿到
    主 run.py 里定义的 get_runner_class (依然在主 src 路径下, 但 import 时
    跑 *子 venv 里的 Python* 解释器执行).

输出协议 (与 run.py:run_system_isolated 配对)
--------------------------------------------
正常结束:
  ...(各种 system 日志输出)...
  __WORKER_RESULT__{"Q1": {...}, "Q2": {...}}__END_WORKER_RESULT__
异常:
  ...(traceback)...
  __WORKER_ERROR__<exc.message>__END_WORKER_ERROR__
  (exit code 1)
两个 marker 都是 stdout 单行字符串, 主进程用 re.search 解析.

引用: [LOG_STRUCTURE.md §5.1 入口与编排](../LOG_STRUCTURE.md)
"""

import argparse
import json
import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv


# ============================================================================
# main — subprocess 入口: 解析 args → 实例化 runner → 跑 queries → 序列化结果
# ============================================================================
# 步骤:
#   1. load_dotenv() — 子进程也要自己读 .env (env var 不会跨 fork 自动继承,
#      因为 subprocess.run 用 fork+exec, 新 Python 进程 os.environ 是空的
#      + 默认继承父进程的 env). 这里多一次 load_dotenv 是防御性写法.
#   2. argparse — 与 run.py 的 worker spawn argv 完全对齐.
#   3. *延迟* import (line 53) `from run import get_runner_class,
#      parse_query_ids`. 必须在 sys.path.insert 之后, 不能 hoist 到顶部
#      (否则 import 失败).
#   4. 实例化 runner_class, 调 run_all_queries() 拿到
#      {query_id: GenericQueryMetric, ...}.
#   5. 序列化成 {"Q{id}": metric_dict} 通过 stdout marker emit.
#
# 错误处理:
#   - 任何 exception 走 except, 打 traceback 到 stderr/stdout, 再 emit
#     __WORKER_ERROR__ marker + exit code 1, 让父进程 (run_system_isolated)
#     能区分 "worker crashed" vs "queries ran but failed".
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run a single system's queries in an isolated environment"
    )
    parser.add_argument(
        "--system", required=True, help="System name (e.g., lotus, palimpzest)"
    )
    parser.add_argument(
        "--use-case", required=True, help="Use case name (e.g., movie, ecomm)"
    )
    parser.add_argument(
        "--queries", nargs="+", default=None, help="Query IDs to run"
    )
    parser.add_argument(
        "--model", default="gemini-2.5-flash", help="Model name"
    )
    parser.add_argument(
        "--scale-factor", type=int, default=None, help="Dataset scale factor"
    )
    parser.add_argument(
        "--skip-setup", action="store_true", help="Skip data setup phase"
    )

    args = parser.parse_args()

    # Import runner infrastructure
    from run import get_runner_class, parse_query_ids

    # Parse query IDs
    query_ids = None
    if args.queries:
        query_ids = parse_query_ids(args.queries)

    # Get runner class
    runner_class = get_runner_class(args.system, args.use_case)
    if not runner_class:
        print(f"Failed to import runner for {args.system}")
        sys.exit(1)

    # Initialize and run
    try:
        runner = runner_class(
            use_case=args.use_case,
            scale_factor=args.scale_factor,
            skip_setup=args.skip_setup,
            model_name=args.model,
        )
        metrics = runner.run_all_queries(queries=query_ids)

        # Write a completion marker with summary for the parent process
        summary = {}
        for query_id, metric in metrics.items():
            summary[f"Q{query_id}"] = metric.to_dict()

        # Print summary as JSON to stdout for parent process
        print(f"\n__WORKER_RESULT__{json.dumps(summary)}__END_WORKER_RESULT__")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\n__WORKER_ERROR__{str(e)}__END_WORKER_ERROR__")
        sys.exit(1)


if __name__ == "__main__":
    main()
