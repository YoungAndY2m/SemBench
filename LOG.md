# SemBench — 修改与配置日志

> 此文件记录 `Desktop/SemBench/` 自 clone 后的全部重要修改与配置变更，供未来回溯参考。
> 三件套之一，与 [LOG_EXEC.md](LOG_EXEC.md)（执行命令手册）和 [LOG_STRUCTURE.md](LOG_STRUCTURE.md)（代码结构说明）配套阅读。

---

## 元数据

- **Fork 来源**: `YoungAndY2m/SemBench` (forked from `SemBench/SemBench`)
- **GitHub origin**: https://github.com/YoungAndY2m/SemBench.git
- **GitHub upstream**: https://github.com/SemBench/SemBench.git
- **Clone 时间**: 2026-05-18（本地拷贝时间）
- **本次记录时 HEAD**: `9bc031f` — "Merge pull request #22 from SemBench/azimmerer/fix-ecomm-lotus-q11-input"
- **Paper**: [Lao et al., SemBench: A Benchmark for Semantic Query Processing Engines](https://arxiv.org/abs/2511.01716) (VLDB 2026)
- **本地 PDF**: [Lao 等 - 2025 - SemBench A Benchmark for Semantic Query Processing Engines.pdf](Lao%20%E7%AD%89%20-%202025%20-%20SemBench%20A%20Benchmark%20for%20Semantic%20Query%20Processing%20Engines.pdf)
- **关联工作计划**:
  - [[Workplan/week-3]] (latest week) —— 当前未明确将 SemBench 列入 PLAN，仅作为后续 ARELY → semantic DB 迁移的 reconnaissance
  - [[Workplan/ALL MODELS]] / [[Workplan/THOUGHT]]
- **关联仓库**:
  - [Desktop/AreCELearnedYet/](../AreCELearnedYet/) —— CE benchmark 主战场，**计划迁移到 SemBench 上做 semantic operator 的 selectivity 估计**
  - [Desktop/AllSQPE/](../AllSQPE/) —— SemBench 评测对象 SQPE 系统的本地副本合集（LOTUS / Palimpzest / ThalamusDB / CAESURA / FlockMTL / Sema / Unify）
- **配套文件**: [LOG_EXEC.md](LOG_EXEC.md)（执行命令手册）, [LOG_STRUCTURE.md](LOG_STRUCTURE.md)（代码结构说明）

---

## 修改记录

> 最新在最上面（changelog 风格）。

### [2026-05-20] 教学注释 pass 完整完成 — 15 个文件 (顶层 cross-cutting Python)

- **目的**: 用户在 session 4 末决定将 SemBench 注释任务列为 #1 (覆盖之前 "必须先做 SQPE 才能 SemBench" 的 prereq), 对 SemBench 顶层 cross-cutting Python 工具做全面教学注释, 配合 [LOG_STRUCTURE.md](LOG_STRUCTURE.md) 让零基础工作者只读注释 + paper 就能扩展 / 改造 SemBench.
- **范围 (用户 AskUserQuestion 选项 3 "全 cross-cutting Python")**: 15 文件, source LoC ~12,033, 注释 +1875 行 (含 file docstring 扩展 + class banner + method banner)
- **修改文件 (全部按 [CLAUDE.md §5.5 §D 硬约束] 0 statement 修改 / 0 行号偏移 / 0 原 docstring 改动, 纯 byte-additive)**:

  **P0 entry layer (2 files +271)**:
  - [src/run.py](src/run.py) — 534 → 751 (+217) — argparse + dispatcher + isolated/direct mode 切换
  - [src/run_worker.py](src/run_worker.py) — 93 → 147 (+54) — subprocess worker + stdout marker 协议

  **P0 scenarios (6 files +495)**:
  - [src/scenario/cars/cars_scenario.py](src/scenario/cars/cars_scenario.py) — 92 → 182 (+90) — ScenarioHandler 抽象基线 + cars 特殊性
  - [src/scenario/medical/medical_scenario.py](src/scenario/medical/medical_scenario.py) — 97 → 151 (+54) — snowflake 系统集成
  - [src/scenario/mmqa/mmqa_scenario.py](src/scenario/mmqa/mmqa_scenario.py) — 134 → 218 (+84) — MMQADataGenerator class + multi-modal
  - [src/scenario/animals/animals_scenario.py](src/scenario/animals/animals_scenario.py) — 165 → 240 (+75) — adversarial data crafting (_ensure_*_pattern)
  - [src/scenario/movie/movie_scenario.py](src/scenario/movie/movie_scenario.py) — 181 → 244 (+63) — review 文本清洗 + caesura 集成
  - [src/scenario/ecomm/ecomm_scenario.py](src/scenario/ecomm/ecomm_scenario.py) — 200 → 329 (+129) — TOML-driven 声明式 query + DuckDB GT

  **P1 base classes (2 files +572)**:
  - [src/runner/generic_runner.py](src/runner/generic_runner.py) — 342 → 590 (+248) — GenericRunner ABC + GenericQueryMetric dataclass + template method pattern
  - [src/evaluator/generic_evaluator.py](src/evaluator/generic_evaluator.py) — 711 → 1035 (+324) — GenericEvaluator ABC + 5 metric dataclass + retrieval/aggregation/ranking/clustering 4 大评估范式

  **P1 visualization & analysis (5 files +537, compact 策略)**:
  - [src/plot.py](src/plot.py) — 4608 → 4780 (+172) — BenchmarkPlotter 32 methods (file + class + 6 method banners)
  - [src/table_brick_design_avg.py](src/table_brick_design_avg.py) — 1160 → 1247 (+87) — LaTeX heatmap table with ±σ
  - [src/aggregate_table_generator.py](src/aggregate_table_generator.py) — 996 → 1082 (+86) — operator × system Q/L + Q/M ratio table
  - [src/plot_scalability_combined.py](src/plot_scalability_combined.py) — 674 → 744 (+70) — GridSpec multi-cell scalability + memory figure
  - [scripts/analysis.py](scripts/analysis.py) — 2046 → 2168 (+122) — SystemAnalyzer 30+ methods, tolerance-relaxed winner analysis

- **风格规范** (按 [CLAUDE.md §5.5 §B](../CLAUDE.md)):
  - 中英夹杂: 概念名 / API / 数学符号保留英文, 解释中文
  - 禁 LaTeX (注释 IDE 渲染不了)
  - 零基础读者目标: 每个 ML/DL/SQPE/统计学概念 (precision / recall / F1 / Spearman / ARI / Omega / pareto frontier / tolerance ε / @dataclass / ABC / template method / matplotlib GridSpec / TOML / DuckDB file_search_path) 第一次出现都给中文释义
  - 大文件 (plot.py / analysis.py / 3 table 工具) 采 compact (file + class + 主要 method) 策略, 不逐方法详注; 中小文件 (run.py / generic_runner / generic_evaluator) 详注

- **硬约束验证全部通过** (按 [CLAUDE.md §5.5 §D](../CLAUDE.md)):
  - 所有 15 文件 `python3 -c "import ast; ast.parse(open('FILE').read())"` ✓
  - 所有 15 文件 `git diff --unified=0 -- FILE | grep -E '^-[^-]'` 返回 **0 deletion** (纯 byte-additive, 无 statement / 缩进 / 行号 / 原 docstring 改动)
  - 总 git stat: `15 files changed, 1875 insertions(+)`
  - Edit 工具 trailing-whitespace pitfall 零事故 (vs CAESURA L1 注释 pass 7 文件中 5 文件中招)

- **实战教训** (从 ANNOTATION_CHECKPOINT.md 实战教训段整理, 现归并到本 changelog):

  1. **LOG_STRUCTURE.md 与实际 repo 小漂移**: scenario 实际是 6 个 (handoff 说 4 个); `run.py` / `plot.py` 在 `src/` 下; `src/result/` 实际是 `src/evaluator/`. 全部按实际位置注释.
  2. **复杂度梯度**: 6 个 scenario 不是 copy-paste, 复杂度从 cars (最简) → medical → mmqa → animals → movie → ecomm (最 elaborate, TOML 声明式). 每个 scenario 按特殊点单独 surface.
  3. **Surface (不修) 的 bugs**:
     - `cars_scenario.py:17` docstring 写 "Medical scenario handler" (copy from medical).
     - `mmqa_scenario.py:58` `FlockMTLMMQASetup` 缺括号.
     - `mmqa_scenario.py` 大小写 q vs Q 不一致 (line 82 vs 124).
     - `generic_evaluator.py:602` `raise "..."` (字符串不是 Exception, 真跑会 TypeError).
     - `compute_*` 7 个 static helpers 都缺 @staticmethod 装饰.
     - `compute_adjusted_rand_index` docstring 说 "group" 列但代码用 "category".
  4. **Compact 策略适用边界**: 当一个 class 有 20+ method 且方法间结构高度同质 (e.g. plot.py 的各 plot_X / analysis.py 的各 calculate_X / find_winners_X), class-level banner 列出方法分组 + 3-5 个关键 method 详注 比逐方法详注更有教学价值 (减少噪音 + 强化整体 mental model). 关键: 必须在 class-level banner 里 *列全* method, 让读者能从一个位置 catalog 所有方法.
  5. **Edit 工具 whitespace 陷阱无事故的原因**: 全程只用 "整段插入" 模式 (Edit 把 `<context>` 替换为 `<context> + <new banner content>`), 不做单行替换; 对 file docstring 严格用 "在原 docstring 内追加新段" 而非整段重写.

- **配套文件状态**:
  - [LOG.md](LOG.md) (本文件) — 追加本条 changelog
  - [LOG_STRUCTURE.md](LOG_STRUCTURE.md) — 不动; 注释里多处引用其 §4.1 / §4.3 / §5.1 / §5.3 / §5.5 / §7.1 / §7.2 章节
  - [LOG_EXEC.md](LOG_EXEC.md) — 不动 (本次纯文档, 没新增执行步骤)
  - `ANNOTATION_CHECKPOINT.md` — 归档到本 changelog 后 `rm` 删除 (按 [CLAUDE.md §5.5 §H.4](../CLAUDE.md))

### [2026-05-18] 初次代码全面扫码 + LOG 三件套创建（仅文档，未改任何代码）

- **目的**: 用户计划将 [Desktop/AreCELearnedYet/](../AreCELearnedYet/) 的 cardinality estimation 工作迁移到 semantic database 场景；本次只做 reconnaissance —— 在不动 upstream 代码的前提下，对 SemBench 全部 src/ + files/ + scripts/ + docs/ 做一次彻底通读，产出可供未来扩展的代码结构地图。
- **修改文件**: 无
- **新增文件**:
  - [LOG.md](LOG.md) —— 本文件
  - [LOG_EXEC.md](LOG_EXEC.md) —— 复制即可用的命令手册（env setup → run → evaluate → plot）
  - [LOG_STRUCTURE.md](LOG_STRUCTURE.md) —— `GenericRunner` / `GenericEvaluator` / scenario handler 三大抽象的接口说明 + paper § ↔ code 映射
- **依据**:
  - submit.html 的 [Implementation Guide section](docs/submit.html) (`#implementation-guide`)
  - [README.md](README.md)
  - [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
  - 全部 `src/` 源码（重点 [src/runner/generic_runner.py](src/runner/generic_runner.py), [src/run.py](src/run.py), [src/run_worker.py](src/run_worker.py), [src/evaluator/generic_evaluator.py](src/evaluator/generic_evaluator.py)）
- **影响**: 无 —— 本次纯 doc-only commit-候选，未触碰 upstream 代码或测试。
- **遗留问题 / 待解明事项** (按优先级):
  1. **submit.html 的 System × Scenario Matrix 与代码实际不一致**:
     - submit.html 矩阵只列 4 系统（LOTUS / Palimpzest / BigQuery / ThalamusDB）× 6 scenarios（含 Medical）
     - 但 [src/run.py:39-47](src/run.py#L39-L47) 的 `get_runner_class` 已支持 7 系统（多出 `snowflake`、`flockmtl`、`caesura`）
     - [src/scenario/](src/scenario/) 下确实有 `caesura_runner`、`flockmtl_runner` 的子目录（每个 scenario 都不全）
     - 看起来 README / submit.html 写于 paper 投稿时点，代码已超前 → 未来若把自己系统加进去，应同时更新 submit.html 矩阵
  2. **README.md 的 5-scenario 表格 vs 实际 6 scenarios**:
     - README §🌟Overview 表只列 Movie / Wildlife / E-Commerce / MMQA / Cars（5 个，无 Medical）
     - 但 [src/scenario/medical/](src/scenario/medical/) 完整实现了 LOTUS / Palimpzest / ThalamusDB / BigQuery / FlockMTL 5 个 runner
     - submit.html 的矩阵也明确把 Medical 列为 6th scenario
     - → Medical 是真实场景，README 文档没跟上
  3. **`get_evaluator` 在 [src/run.py:79-87](src/run.py#L79-L87) 引用了一个 `detective` use case**, 但 `src/scenario/` 下无 `detective/` 目录 → dead reference，未来加 use case 时需要清理。
  4. **`detective` 在 [src/runner/generic_runner.py:335-336](src/runner/generic_runner.py#L335-L336) 的 `get_scenario_handler` 里被显式 return `None`** —— 看起来曾经存在但已被废弃，与上一条印证。
  5. **`scale_factor` 含义跨 scenario 不一致**:
     - Movie scenario：默认 2000，表示采样的 review 行数（[src/scenario/movie/movie_scenario.py:20](src/scenario/movie/movie_scenario.py#L20))
     - Cars scenario：[files/cars/query/lotus/Q1.py:4](files/cars/query/lotus/Q1.py#L4) 默认 `scale_factor=157376` —— 看起来是 complaints 表的总行数
     - 即 scale_factor 是 per-scenario 自定义维度，**不是统一含义**；README "each use case has its own range" 是字面真的
  6. **CE 切入点定位**（与 ARELY 迁移直接相关）:
     - 每个 `_execute_q*()` / `Q{i}.py` 内部的 `sem_filter` / `sem_join` / `sem_map` 调用是潜在的"语义算子 selectivity 估计"埋点
     - [src/evaluator/generic_evaluator.py](src/evaluator/generic_evaluator.py) 的 `_generic_retrieval_evaluation` 用 ground truth set 比较 → 中间步骤的真实通过行数没有显式保存，但可以从 `_get_ground_truth` 走 DuckDB 的 gold SQL ([src/scenario/movie/evaluation/evaluate.py:36-60](src/scenario/movie/evaluation/evaluate.py#L36-L60)) 推出
     - → ground truth 行数可拿到，**CE 模型的监督信号是足够的**

---

## 当前环境配置

- **Python**: 3.12（由 [scripts/setup_envs.sh:33](scripts/setup_envs.sh#L33) 强制；通过 uv 装隔离 venv）
- **包管理器**: [uv](https://docs.astral.sh/uv/)（脚本自动安装到 `~/.local/bin`）
- **多 venv 隔离**（避免 `numpy<2` vs `numpy>=2` 冲突）：
  - `.venvs/sembench/` —— orchestrator，跑 `run.py` / 评测 / 画图（依赖 [requirements/base.txt](requirements/base.txt)）
  - `.venvs/lotus/` —— `lotus-ai==1.1.3` + `chromadb` + `faiss-cpu`（依赖 [requirements/lotus.txt](requirements/lotus.txt)）
  - `.venvs/palimpzest/` —— `palimpzest @ git+...@0.8.2.sem_agg`
  - `.venvs/thalamusdb/` —— `thalamusdb==0.1.15` (`from tdb.data.relational import Database`)
  - `.venvs/bigquery/` —— `google-cloud-bigquery`
  - `.venvs/caesura/` —— CAESURA
  - `.venvs/flockmtl/` —— DuckDB FlockMTL extension
- **GPU 处理**: [scripts/setup_envs.sh:107-114](scripts/setup_envs.sh#L107-L114) 自动检测 `nvidia-smi` → 装 CUDA torch；否则装 CPU torch（cu124 / cpu wheel index）
- **数据**:
  - Movie 测试数据已经包含 `files/movie/data/sf_2000/{Movies,Reviews}.csv` —— README 强调这是 demo only，做实验前应 `rm -rf files/{scenario}/data/` 让 `setup_scenario()` 重新从 Google Drive 拉取（[src/scenario/movie/movie_scenario.py:46-87](src/scenario/movie/movie_scenario.py#L46-L87)）
  - 数据集 Google Drive: https://drive.google.com/drive/folders/1pqf8DKFai16MR80Z7pcls5FgBbom-IJt
- **`.env` / 环境变量**（来自 [.env.example](.env.example)）:
  - `OPENAI_ORGANIZATION`, `OPENAI_API_KEY`
  - `GOOGLE_APPLICATION_CREDENTIALS`, `GCLOUD_PROJECT`（BigQuery 必需）
  - `KAGGLE_USERNAME`, `KAGGLE_KEY`（可选，部分数据集走 Kaggle）

---

## 已知 pitfall

- **`requirements.txt` 与 `requirements/` 不是一回事**：根目录的 [requirements.txt](requirements.txt) 把 `lotus-ai==1.1.3`、`palimpzest==0.8.2`、`thalamusdb==0.1.15` 装在同一环境里 —— 这恰好是 [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) 警告会冲突的方式（numpy 版本不兼容）。**不要直接 `pip install -r requirements.txt`**；应使用 `bash scripts/setup_envs.sh` 走 uv 多 venv 流程。
- **`run.py` 在结尾调用 `os._exit(0)`**（[src/run.py:530](src/run.py#L530)）—— 这是为了强制结束 LOTUS 后台连接池线程，意味着 `run.py` 永远不会 `return` 给调用者，IPython / Jupyter / pytest 里不能用。
- **`run_worker.py` 的标记协议靠正则解析 stdout**（`__WORKER_RESULT__...__END_WORKER_RESULT__`）—— 见 [src/run.py:201-216](src/run.py#L201-L216)。若 worker 自己向 stdout 输出含这些字符串的日志，会破坏解析。新增系统时要避免。
- **`models.toml` 文件存在于 [src/](src/) 根**但没看到代码引用，疑似 todo / future feature。新加模型时不要假设它生效。
- **BigQuery cost 取数有 5s 等待 + 3 次重试**（[src/runner/generic_bigquery_runner/generic_bigquery_runner.py:186-282](src/runner/generic_bigquery_runner/generic_bigquery_runner.py#L186-L282)）——  这是因为 `inference_logs` 表 materialization 有延迟。`bigquery` 子任务跑完后总是会有 5-15s 额外等待，不是 hang。
- **`thalamusdb` 走的 `tdb` 包**（不是 `thalamusdb`）：[generic_thalamusdb_runner.py:23-26](src/runner/generic_thalamusdb_runner/generic_thalamusdb_runner.py#L23-L26) 显示从 pip 装的包 import 名是 `tdb`，文件命名混乱时容易踩。
