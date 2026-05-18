# SemBench — 代码结构说明（LOG_STRUCTURE）

> 配合 paper [Lao et al., SemBench, arXiv:2511.01716](https://arxiv.org/abs/2511.01716)（本地副本 [Lao 等 - 2025 - SemBench.pdf](Lao%20%E7%AD%89%20-%202025%20-%20SemBench%20A%20Benchmark%20for%20Semantic%20Query%20Processing%20Engines.pdf)）+ 官方 submit.html `#implementation-guide` 一起读。
> 三件套之一，与 [LOG.md](LOG.md)（"经历了什么修改"）和 [LOG_EXEC.md](LOG_EXEC.md)（"怎么跑"）配套。
> 目标读者：零基础工作者通过 "paper + 本文件" 就能理解每个文件干什么、扩展点在哪。

---

## 0. 元数据

- **Paper / Spec**: Lao et al., *SemBench: A Benchmark for Semantic Query Processing Engines*, VLDB 2026 — [arXiv:2511.01716](https://arxiv.org/abs/2511.01716)
- **核心问题**: 把"用 LLM 语义算子（sem_filter/sem_join/sem_map/sem_rank/sem_classify）查询多模态数据"这件事变成可比较、可复现的 benchmark；同时评测 4-7 个真实 SQPE（Semantic Query Processing Engine）。
- **方法分类总图**:
  - **Code 模式**: Python 库式系统（LOTUS、Palimpzest）——每条 query 是一个 `_execute_q{i}()` Python 方法
  - **Code\* 模式**: 同样是 Python，但 query 代码外置到 `files/{scenario}/query/{system}/Q{i}.py`，runner 动态 import
  - **SQL 模式**: SQL 引擎（BigQuery）——每条 query 是 `Q{i}.sql` 模板文件，runtime 用 Jinja2 替换 `<<variable>>`
  - **Hybrid 模式**: ThalamusDB ——既支持 SQL 文件也支持 `_execute_q*()`，runtime 自动 fallback

---

## 1. Repo 顶层布局

```
SemBench/
├── README.md                         # 项目门面
├── ENVIRONMENT_SETUP.md              # uv 多 venv 详细说明
├── requirements.txt                  # 单 venv 全量依赖（与多 venv 流冲突，慎用）
├── requirements/                     # 多 venv 流的分包依赖
│   ├── base.txt                      # 共享框架依赖（pandas/torch/litellm 等）
│   ├── lotus.txt                     # -r base.txt + lotus-ai==1.1.3
│   ├── palimpzest.txt                # palimpzest @ git+...@0.8.2.sem_agg
│   ├── thalamusdb.txt                # thalamusdb==0.1.15
│   ├── bigquery.txt                  # google-cloud-bigquery
│   ├── caesura.txt                   # CAESURA
│   └── flockmtl.txt                  # DuckDB FlockMTL extension
├── scripts/
│   ├── setup_envs.sh                 # 一键装 venv（uv 驱动，按需选系统）
│   ├── repeat_experiment.sh          # 多次跑同一个 config 拿 error bar
│   ├── scale_factor_experiment_*.sh  # 跑 scale-factor sweep
│   ├── evaluate_palimpzest.sh        # palimpzest 专用评测脚本
│   ├── analysis.py                   # 跨 system / scenario 分析报告
│   ├── deploy-website.sh             # docs/ 部署到 GitHub Pages
│   └── stop-website.sh
├── config/
│   └── system/
│       ├── palimpzest/               # 10 个 JSON：{model}-{policy} 组合配置
│       └── thalamusdb/               # 模型配置 JSON + models.json
├── docs/                             # GitHub Pages 静态站
│   ├── index.html
│   ├── submit.html                   # ← Implementation Guide 在这里
│   ├── static/                       # CSS / JS / 图片
│   ├── sitemap.xml
│   └── robots.txt
├── files/                            # 数据 + 查询 + 结果（按 scenario 组织）
│   └── {scenario}/                   # {animals, cars, ecomm, medical, mmqa, movie}
│       ├── data/sf_{scale_factor}/   # 实际数据库文件（CSV / 图片 / 音频）
│       ├── query/
│       │   ├── natural_language/Q{i}.txt   # NL 查询描述（人读）
│       │   ├── gold_sql/Q{i}.sql           # 评测 ground truth 用的 DuckDB SQL
│       │   ├── {system}/Q{i}.sql 或 .py    # 各 system 的具体查询实现（SQL/Code* 模式）
│       │   └── coverage.json               # 该 scenario 各 system 已实现的 query 清单
│       ├── raw_results/{system}/Q{i}.csv   # 每次跑 runner 写到这里
│       └── metrics/{system}.json           # evaluator 写到这里（精度/F1/relative_error）
├── src/                              # 全部 Python 代码
│   ├── run.py                        # CLI 入口，dispatcher
│   ├── run_worker.py                 # 隔离 venv 内的 subprocess worker
│   ├── plot.py                       # 出 bar chart / pareto figure
│   ├── plot_scalability_combined.py  # scale-factor sweep 图
│   ├── temp_plot_join.py             # paper 用 join 实验图（临时脚本）
│   ├── temp_plot_reasoning.py        # paper 用 reasoning 实验图（临时脚本）
│   ├── table_brick_design_avg.py     # paper 用 LaTeX 大表（avg 版）
│   ├── aggregate_table_generator.py  # 跨 scenario 汇总表
│   ├── models.toml                   # 模型常量表（疑似 todo，未见 code 引用）
│   ├── evaluator/
│   │   └── generic_evaluator.py      # GenericEvaluator 基类 + 通用 metric 计算
│   ├── runner/
│   │   ├── generic_runner.py         # ★★ 全 system 的根基类（必读）
│   │   ├── generic_lotus_runner/
│   │   │   └── generic_lotus_runner.py    # LOTUS 中间基类
│   │   ├── generic_palimpzest_runner/
│   │   │   └── generic_palimpzest_runner.py
│   │   ├── generic_bigquery_runner/
│   │   │   └── generic_bigquery_runner.py
│   │   ├── generic_thalamusdb_runner/
│   │   │   ├── generic_thalamusdb_runner.py
│   │   │   └── config/                    # ThalamusDB 模型配置
│   │   ├── generic_flockmtl_runner/
│   │   ├── generic_caesura_runner/
│   │   │   ├── generic_caesura_runner.py
│   │   │   └── caesura/                   # vendored CAESURA 源码
│   │   └── (各自只有 base, 没有 __init__.py)
│   └── scenario/                     # 每个 scenario 独立模块
│       └── {scenario}/               # {animals, cars, ecomm, medical, mmqa, movie}
│           ├── {scenario}_scenario.py     # ScenarioHandler：数据下载/构建/分发
│           ├── README.md
│           ├── preparation/
│           │   └── generate_data.py       # 从 Google Drive / Kaggle 拉数据 + 采样构造
│           ├── setup/
│           │   ├── bigquery.py            # 把数据 load 到 BigQuery dataset
│           │   └── flockmtl.py            # 把数据 load 到 DuckDB
│           ├── runner/
│           │   ├── lotus_runner/{system}_runner.py    # 继承 GenericLotusRunner
│           │   ├── palimpzest_runner/
│           │   ├── bigquery_runner/
│           │   ├── thalamusdb_runner/
│           │   ├── flockmtl_runner/
│           │   └── caesura_runner/         （仅 movie scenario 有）
│           └── evaluation/
│               └── evaluate.py            # 继承 GenericEvaluator
├── figures/                          # 跑完 plot.py 后的 PNG 输出
├── analysis_results/                 # 跑完 scripts/analysis.py 后的分析 CSV
├── assets/                           # logo 等静态资源
├── LICENSE-APACHE / LICENSE-MIT      # 双许可
├── .env.example                      # 环境变量模板
├── .vscode/                          # IDE 配置
└── .git/, .gitignore                 # （含 .venvs/ 排除）
```

---

## 2. 端到端 Pipeline

```
                                     ┌─────────────────────────────────────┐
   user CLI                          │  Google Drive datasets              │
   $ python src/run.py               │  https://drive.google.com/...       │
   --systems lotus                   └────────────────┬────────────────────┘
   --use-cases movie                                  │ first run only
   --queries 1                                        ▼
   --model gemini-2.5-flash                ┌───────────────────────┐
   --scale-factor 2000                     │ scenario_handler      │
        │                                  │ .setup_scenario(...)  │── samples ──►  files/{scenario}/data/sf_N/*.csv
        ▼                                  └─────────┬─────────────┘
   src/run.py:main()                                  │
        │                                             │
        │ (1) parse args → query_ids                  │
        │ (2) detect .venvs/{system}/                 │
        │                                             │
        │   [Isolated path]                           │
        │   spawn subprocess via                      │
        │   .venvs/lotus/bin/python                   │
        │     src/run_worker.py                       │
        ▼                                             │
   src/run_worker.py                                  │
        │                                             │
        │ get_runner_class(system, use_case)          │
        │   → import scenario.{uc}.runner.            │
        │     {sys}_runner.{sys}_runner               │
        │                                             │
        │ RunnerClass(...)._initialize_   ◄───────────┘
        │   (calls scenario_handler.setup_scenario
        │    unless skip_setup)
        │
        ▼
   ── for each query_id ───────────────────────────────────────
   │                                                          │
   │   GenericLotusRunner.execute_query(qid)                  │
   │   ├─► _discover_query_impl(qid)                          │
   │   │     # finds method "_execute_q{qid}" via reflection  │
   │   │                                                      │
   │   └─► LotusRunner._execute_q1():                         │
   │         reviews = self.load_data("Reviews.csv")          │
   │         reviews.sem_filter('...')                        │
   │         return DataFrame                                 │
   │                                                          │
   │   metric.results = DataFrame                             │
   │   metric.token_usage = lotus.settings.lm.stats...        │
   │   metric.money_cost = _calculate_cost(...)               │
   │                                                          │
   ├─► run_worker prints __WORKER_RESULT__{json}__END...      │
   │                                                          │
   ◄── back to src/run.py ────────────────────────────────────
        │
        │ save_metrics()  → files/{sc}/metrics/{sys}.json
        │ save_results()  → files/{sc}/raw_results/{sys}/Q{i}.csv
        │
        ▼
   evaluator_class = get_evaluator(use_case)
   evaluator.evaluate_system(system_name)
        │
        │ for each query:
        │   sys_df = pd.read_csv(raw_results/{sys}/Q{i}.csv)
        │   gt_df  = _get_ground_truth(qid)
        │             └── runs gold SQL via duckdb on data/sf_N/
        │   metric = _evaluate_single_query(qid, sys_df, gt_df)
        │             └── _generic_retrieval / _aggregation / _ranking
        │
        ▼
   files/{sc}/metrics/{sys}.json   (extended with precision/recall/F1/...)

   ────────────────────────────────────────────────────────────

   src/plot.py            → figures/{sc}/*.png  (bar / pareto)
   src/table_brick_design_avg.py → LaTeX 大表
   scripts/analysis.py    → analysis_results/*.csv
```

---

## 3. Paper 章节 → 代码文件映射

| Paper § | 对应 SemBench 代码 | 责任 |
|---------|-------------------|------|
| §1 Intro（什么是 semantic operator） | [src/scenario/movie/runner/lotus_runner/lotus_runner.py:82](src/scenario/movie/runner/lotus_runner/lotus_runner.py#L82) `sem_filter` 调用、`sem_map` 调用、`sem_join` 调用、`sem_topk` 调用 | concrete operator 示例都在 movie scenario 里 |
| §2 Benchmark Design（diversity in scenarios/modalities/operators） | [files/{scenario}/](files/) 的 6 个目录 + [README.md](README.md) §🌟Overview 表 | scenario 矩阵 + modality 覆盖 |
| §3 Systems Evaluated | [src/runner/generic_*_runner/](src/runner/) 下 6 个系统的 base runner | 每个系统一个中间基类 |
| §4.1 Code mode | [src/runner/generic_lotus_runner/](src/runner/generic_lotus_runner/) + [src/scenario/movie/runner/lotus_runner/lotus_runner.py](src/scenario/movie/runner/lotus_runner/lotus_runner.py) | LOTUS / Palimpzest |
| §4.2 SQL mode | [src/runner/generic_bigquery_runner/](src/runner/generic_bigquery_runner/) + [files/movie/query/bigquery/Q1.sql](files/movie/query/bigquery/Q1.sql) | BigQuery |
| §4.3 Code* mode | [files/cars/query/lotus/Q1.py](files/cars/query/lotus/Q1.py) + [src/scenario/cars/runner/lotus_runner/lotus_runner.py](src/scenario/cars/runner/lotus_runner/lotus_runner.py)（minimal） | 新场景的外置 Python 查询 |
| §4.4 Hybrid | [src/runner/generic_thalamusdb_runner/generic_thalamusdb_runner.py:96-110](src/runner/generic_thalamusdb_runner/generic_thalamusdb_runner.py#L96-L110) | ThalamusDB 先试 SQL 文件，找不到 fallback 到 `_execute_q*` |
| §5.1 Quality Metrics（P/R/F1/relative error/rank corr） | [src/evaluator/generic_evaluator.py:30-67](src/evaluator/generic_evaluator.py#L30-L67) 的 `QueryMetric*` 数据类 + `_generic_*_evaluation` 方法 | 不同 query 类型不同 metric |
| §5.2 Cost / Latency / Tokens | [src/runner/generic_runner.py:21-44](src/runner/generic_runner.py#L21-L44) 的 `GenericQueryMetric` 字段 + 各 system 的 `_update_token_usage()` | 三类成本指标 |
| §5.3 Per-query 1-hour timeout | 未在代码里强制，靠 `concurrent_llm_worker` + LLM provider rate limit 间接实现 | -- |
| §6 Findings | 这部分没有对应 code，只在 paper / [figures/](figures/) 体现 | -- |

---

## 4. 核心抽象

> 给"未来要扩展 / 接系统"的人看的接口契约。所有 4 个抽象都按 `先字段 → 再方法 → 关键 hook` 列出。

### 4.1 `GenericRunner`（[src/runner/generic_runner.py](src/runner/generic_runner.py)）

**字段**（在 `__init__` 里建好；扩展者一般不动）:

| 字段 | 类型 | 来源 |
|------|------|------|
| `use_case` | `str` | 构造参数 |
| `scale_factor` | `int` | 构造参数 |
| `model_name` | `str` | 构造参数 |
| `concurrent_llm_worker` | `int` | 构造参数（默认 20） |
| `system_name` | `str` | `self.get_system_name()` —— 子类实现 |
| `base_path` | `Path` | `src/runner/generic_runner.py` 上溯两级 → repo 根 |
| `files_path` | `Path` | `{base}/files/{use_case}/` |
| `data_path` | `Path` | `{files}/data/sf_{scale_factor}/` |
| `query_path` | `Path` | `{files}/query/` |
| `results_path` | `Path` | `{files}/raw_results/{system_name}/`（自动 mkdir） |
| `metrics_path` | `Path` | `{files}/metrics/`（自动 mkdir） |
| `scenario_handler` | object \| None | `GenericRunner.get_scenario_handler(use_case, scale_factor)` |
| `metrics` | `Dict[int, GenericQueryMetric]` | `run_all_queries()` 填 |

**子类必须实现**:

```python
@abstractmethod
def get_system_name(self) -> str: ...
```

**子类一般覆盖**:

```python
def execute_query(self, query_id: int) -> GenericQueryMetric: ...
# 或者整体接管：
def execute_queries(self, query_ids: List[int]) -> Dict[int, GenericQueryMetric]: ...
```

**框架提供的方法（扩展者可直接调用）**:

| 方法 | 用途 |
|------|------|
| `load_data(filename, **kwargs) -> pd.DataFrame` | 从 `data_path` 读 CSV |
| `get_query_text(qid, query_type="natural_language") -> str` | 读 `query_path/{query_type}/Q{i}.txt` |
| `save_results(qid, df) -> None` | 写 `results_path/Q{i}.csv` |
| `save_metrics() -> None` | 写 `metrics_path/{system}.json` |
| `run_all_queries(queries=None) -> Dict[int, GenericQueryMetric]` | 主循环：discover → execute → save |
| `_discover_queries() -> List[int]` | 默认走 scenario handler / `mm_sql/` |
| `_discover_query_impl(qid) -> callable` | **Code 模式核心**：用 `getattr` 找 `_execute_q{qid}` |
| `_discover_query_text(qid) -> str` | **SQL 模式核心**：读 `query_path/{system_name}/Q{qid}.sql` |

**关键 hook (paper §4.4 Hybrid 模式靠这个)**:

```python
# 文件: src/runner/generic_runner.py:310-342
@staticmethod
def get_scenario_handler(use_case: str, scale_factor: int = None):
    """根据 use_case 字符串动态 import 对应 scenario 模块；返回 ScenarioHandler 实例。"""
```

---

### 4.2 `GenericQueryMetric`（[src/runner/generic_runner.py:21-44](src/runner/generic_runner.py#L21-L44)）

```python
@dataclass
class GenericQueryMetric:
    query_id: int
    status: str                  # 'success' | 'failed' | 'pending'
    execution_time: float = None
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    token_usage: int = None
    money_cost: float = None
    error: Optional[str] = None
```

**特殊**: `to_dict()` 序列化时**不会**把 `results` DataFrame 一起塞进 JSON，只保留 `row_count = len(results)`。原始结果靠 `save_results()` 单独落 CSV。

---

### 4.3 `GenericEvaluator`（[src/evaluator/generic_evaluator.py:94](src/evaluator/generic_evaluator.py#L94)）

**子类必须实现**:

```python
def _load_domain_data(self) -> None: ...                            # 加载 scenario CSV
def _get_ground_truth(self, query_id: int) -> pd.DataFrame: ...     # 通常跑 gold_sql/Q{i}.sql 经 DuckDB
def _evaluate_single_query(self, qid, sys_df, gt_df) -> QueryMetric...:
    ...
    # 通常 dispatch 到 self._evaluate_q{qid}(sys_df, gt_df)
```

**辅助 metric 计算（已实现，开箱即用）**:

| 方法 | 用途 |
|------|------|
| `_generic_retrieval_evaluation` | 集合匹配 → P/R/F1 |
| `_generic_aggregation_evaluation` | 单值数值差异 → absolute_error / relative_error / MAPE |
| `_generic_ranking_evaluation` | 两个评分序列 → Spearman / Kendall tau |
| `_evaluate_tuple_matching(n_columns)` | 多列 tuple 匹配（如 pair join） |
| `_evaluate_unique_values(column_index)` | 单列去重值的 set 匹配 |
| `compute_precision/recall/f1_score` | 按 `id` 列做 set 匹配 |
| `compute_adjusted_rand_index` | 聚类指标（用 `category` 列） |
| `compute_omega_index` | 重叠社区指标（cdlib 实现） |
| `compute_f1_score_classify` | sklearn `f1_score(average='macro')` |

**返回类型**（4 选 1）:

```python
@dataclass class QueryMetricRetrieval:    precision, recall, f1_score
@dataclass class QueryMetricAggregation:  relative_error, absolute_error, MAPE
@dataclass class QueryMetricRank:         spearman_correlation, kendall_tau
@dataclass class SingleAccuracyScore:     accuracy, metric_type
```

**结果落盘**: `files/{scenario}/metrics/{system}.json` —— 同一个文件被 runner 和 evaluator **联合写入**（runner 写 row_count/time/tokens/cost，evaluator 写质量字段，靠 `dict.update()` 合并）。

---

### 4.4 Scenario Handler（如 [src/scenario/movie/movie_scenario.py:11](src/scenario/movie/movie_scenario.py#L11)）

每个 scenario 一份，不强制继承（duck-typed）。需要的方法：

| 方法 | 用途 |
|------|------|
| `__init__(self, scale_factor)` | 持有 scale_factor |
| `setup_scenario(self, systems: List[str]) -> None` | 下数据 + 构造 DB + 把数据 load 到每个系统的存储（BigQuery / DuckDB / 原始 CSV） |
| `get_query_text(self, query_id, system_name) -> str` | 给 Code\* / SQL 模式提供查询文本 |
| `get_data_dir(self) -> str` | 返回 `files/{scenario}/data/sf_{N}/` 的绝对路径 |
| `discover_available_queries(self, system_name=None) -> List[int]` | 列出该 scenario × system 已实现的 query 编号 |

---

## 5. 模块/文件逐一说明

### 5.1 入口与编排

#### [src/run.py](src/run.py)
- **职责**: CLI 入口；解析参数；探测 `.venvs/` 决定是否走 subprocess 隔离；调度 runner；调评测；打印摘要。
- **关键函数**:
  - `get_runner_class(system, use_case) -> RunnerClass` ([src/run.py:35-70](src/run.py#L35-L70)) —— 动态 import `scenario.{uc}.runner.{sys}_runner.{sys}_runner`；返回类名 `{Sys}Runner`
  - `get_evaluator(use_case) -> EvalClass` ([src/run.py:73-111](src/run.py#L73-L111)) —— 动态 import `scenario.{uc}.evaluation.evaluate`
  - `parse_query_ids(args) -> List[int]` ([src/run.py:114-136](src/run.py#L114-L136)) —— 支持 `1`/`Q1` 两种格式
  - `run_system_isolated(...) -> Dict` ([src/run.py:147-229](src/run.py#L147-L229)) —— spawn `.venvs/{sys}/bin/python src/run_worker.py ...`；解析 `__WORKER_RESULT__...__END_WORKER_RESULT__` marker
  - `run_benchmark(...) -> Dict` ([src/run.py:232-344](src/run.py#L232-L344)) —— 主循环，自动 fallback isolated → direct
  - `main()` ([src/run.py:347-530](src/run.py#L347-L530)) —— argparse + summary print + `os._exit(0)`
- **CLI 参数**:
  - `--systems` (nargs="+", default `["lotus"]`)
  - `--use-cases` (nargs="+", default `["movie"]`)
  - `--queries` (nargs="+", default None = 全部)
  - `--model` (default `"gemini-2.5-flash"`)
  - `--scale-factor` (int)
  - `--skip-setup` (action="store_true")
  - `--no-isolation` (action="store_true")
  - `--verbose`（未实际使用）

#### [src/run_worker.py](src/run_worker.py)
- **职责**: 隔离 venv 内的 subprocess worker；只跑单个 system 的所有 query；以 stdout marker 形式把结果回传给父进程。
- **接口**: `--system`, `--use-case`, `--queries`, `--model`, `--scale-factor`, `--skip-setup`
- **stdout 协议**: `__WORKER_RESULT__{json}__END_WORKER_RESULT__` / `__WORKER_ERROR__{msg}__END_WORKER_ERROR__`

### 5.2 Runner 基类层（src/runner/）

#### [src/runner/generic_runner.py](src/runner/generic_runner.py)
- **职责**: 所有 system 的 base class；定义路径约定、`GenericQueryMetric` 数据类、Code/SQL 两种 query 发现机制、scenario handler dispatch。
- **接口**: 见 §4.1。
- **I/O 约定**: 全部由 base class 强制（`data_path` / `results_path` / `metrics_path`）；子类直接用 `self.load_data(name)` 即可。

#### [src/runner/generic_lotus_runner/generic_lotus_runner.py](src/runner/generic_lotus_runner/generic_lotus_runner.py)
- **职责**: 装 LOTUS LM；warmup（带 exp backoff，3 次重试）；token 统计 + cost 计算（按 [PRICING dict](src/runner/generic_lotus_runner/generic_lotus_runner.py#L24-L54) 9 个模型）；`_discover_queries` 用 regex 找所有 `_execute_q\d+` 方法。
- **关键参数**: `policy='approximate'` / `'exact'`, `ranking='map'` / `'topk'`, `max_tokens=8192`
- **接口**: 继承 `GenericRunner`，子类只需实现 `_execute_q*()` 方法即可。
- **I/O**: 读 `data_path` 下 CSV / 图像 / 音频；写 `results_path/Q{i}.csv`。

#### [src/runner/generic_palimpzest_runner/generic_palimpzest_runner.py](src/runner/generic_palimpzest_runner/generic_palimpzest_runner.py)
- **职责**: 装 Palimpzest 的 `QueryProcessorConfig`；支持从 [config/system/palimpzest/{name}.json](config/system/palimpzest/) 读策略（MaxQuality / MinCost、available models、reasoning_effort 等）；token / cost 走 `exec_stats.total_tokens` & `total_execution_cost`。
- **特殊**: 支持环境变量 `PALIMPZEST_CONFIG_FILE` 覆盖构造参数 `config_file`。
- **I/O**: 同上。

#### [src/runner/generic_bigquery_runner/generic_bigquery_runner.py](src/runner/generic_bigquery_runner/generic_bigquery_runner.py)
- **职责**: 装 BQ Client；走 SQL 模式 —— 调 `_discover_query_text()` 拿 `.sql` 模板 → Jinja2 `<<var>>` 替换（`variable_start_string="<<"`, `variable_end_string=">>"`）→ `bq_client.query()` 执行 → 跑后 5s 等 `inference_logs` materialize，再查 token & cost 聚合 SQL。
- **替换的变量**: `<<connection>>`、`<<query_id>>`（带 UUID）、`<<other_params>>`、`<<thinking_budget>>`
- **关键 SQL**: [generic_bigquery_runner.py:133-179](src/runner/generic_bigquery_runner/generic_bigquery_runner.py#L133-L179) 的 `AGG_SQL` —— UNION ALL 5 个 model 的 inference_logs，按 query_uuid 过滤聚合
- **重试**: 最多 3 次 × 5s（共 15s 额外延迟），用于 inference_logs materialization。

#### [src/runner/generic_thalamusdb_runner/generic_thalamusdb_runner.py](src/runner/generic_thalamusdb_runner/generic_thalamusdb_runner.py)
- **职责**: Hybrid 模式 —— 优先 `scenario_handler.get_query_text(qid, "thalamusdb")` 拿 SQL；找不到 fallback `_discover_query_impl(qid)` 找 Python 方法。
- **依赖**: `from tdb.data.relational import Database`（pip 包是 `thalamusdb`，但 import name 是 `tdb`）
- **DB**: 默认 `{data_dir}/thalamusdb.duckdb`
- **限制**: `max_calls=1e11, max_seconds=6000, max_tokens=1e22` —— 实际无限大，等价于不限。

#### [src/runner/generic_flockmtl_runner/generic_flockmtl_runner.py](src/runner/generic_flockmtl_runner/generic_flockmtl_runner.py)
- **职责**: FlockMTL（DuckDB extension）的 base runner。详见文件本身。

#### [src/runner/generic_caesura_runner/generic_caesura_runner.py](src/runner/generic_caesura_runner/generic_caesura_runner.py) + [src/runner/generic_caesura_runner/caesura/](src/runner/generic_caesura_runner/caesura/)
- **特别**: CAESURA 源码被 vendored 在子目录里（不是 pip install），看起来是为了 patch 它的内部接口。

### 5.3 Scenario 层（src/scenario/）

每个 scenario 目录都长这样:

```
src/scenario/{name}/
├── {name}_scenario.py        # ScenarioHandler
├── README.md                 # scenario 介绍
├── preparation/
│   └── generate_data.py      # 从 Google Drive 拉 + 采样 + 落 CSV
├── setup/
│   ├── bigquery.py           # 把数据 load 到 BigQuery
│   └── flockmtl.py           # 把数据 load 到 DuckDB FlockMTL
├── runner/
│   └── {sys}_runner/{sys}_runner.py   # 继承 src/runner/generic_{sys}_runner
└── evaluation/
    └── evaluate.py           # MyEvaluator(GenericEvaluator)
```

#### [src/scenario/movie/](src/scenario/movie/)
- **Modalities**: table + text
- **Data**: `Movies.csv` + `Reviews.csv`，scale_factor = 采样的 review 行数
- **10 queries**: filter / count / ratio / pairwise sentiment join / sentiment classify / movie rank（[src/scenario/movie/runner/lotus_runner/lotus_runner.py:71-597](src/scenario/movie/runner/lotus_runner/lotus_runner.py#L71-L597) 是最全的实现）
- **All 6 systems 都有 runner**（lotus / palimpzest / bigquery / thalamusdb / flockmtl / caesura）

#### [src/scenario/animals/](src/scenario/animals/)
- **Modalities**: table + image + audio
- **8K+ images, 650 audio**
- **10 queries**: 物种识别 / 地理共现 / 跨模态相关
- **5 systems**: lotus / palimpzest / bigquery / thalamusdb / 缺 flockmtl/caesura

#### [src/scenario/ecomm/](src/scenario/ecomm/)
- **Modalities**: table + text + image
- **44K text + 44K images**
- **14 queries**: 含 join / map / rank / classify 各算子最多
- **5 systems**: 同上

#### [src/scenario/mmqa/](src/scenario/mmqa/)
- **Modalities**: table + text + image
- **基于 MultiModalQA 数据集**: 5K text + 1K images
- **11 queries**: filter / join / map 跨模态问答

#### [src/scenario/cars/](src/scenario/cars/)
- **Modalities**: table + text + image + audio（最全）
- **157K text + 30K image + 1.4K audio**
- **10 queries**: 多模态故障诊断
- **Code\* 示例**: [files/cars/query/lotus/Q1.py](files/cars/query/lotus/Q1.py)

#### [src/scenario/medical/](src/scenario/medical/)
- **未在 README 表格里出现，但 submit.html 和 code 都把它列为正式 scenario**
- **5 systems**: lotus / palimpzest / bigquery / thalamusdb / flockmtl

### 5.4 Evaluator 层（src/evaluator/）

#### [src/evaluator/generic_evaluator.py](src/evaluator/generic_evaluator.py)
- **职责**: 抽象 evaluator 基类（见 §4.3）+ 一堆 metric helper
- **结果落盘逻辑** ([src/evaluator/generic_evaluator.py:126-172](src/evaluator/generic_evaluator.py#L126-L172)): 读已存在的 `metrics/{system}.json`，做 `dict.update()` 合并字段，再写回 —— 不会冲掉 runner 已写的 row_count/time/tokens/cost。

### 5.5 可视化与分析

#### [src/plot.py](src/plot.py)
- 按 scenario 出 bar chart（每个 metric 一张）+ pareto 图（cost-quality trade-off）

#### [src/table_brick_design_avg.py](src/table_brick_design_avg.py)
- paper Table 用：所有 system × scenario × metric 的 LaTeX 大表

#### [src/aggregate_table_generator.py](src/aggregate_table_generator.py)
- 跨 scenario 汇总

#### [src/plot_scalability_combined.py](src/plot_scalability_combined.py)
- scale-factor sweep 的趋势图

### 5.6 配置与脚本

#### [config/system/palimpzest/](config/system/palimpzest/)
10 个 JSON 文件，命名 `{model}-{policy}.json` —— 例 `gemini-2.5-flash-mincost.json`。结构：

```json
{
  "policy": "MaxQuality",
  "available_models": ["GEMINI_2_5_FLASH"],
  "execution_strategy": "parallel",
  "max_workers": 20,
  "join_parallelism": 20,
  ...
}
```

#### [config/system/thalamusdb/](config/system/thalamusdb/)
- `gemini_2.5flash.json` / `gemini_2.5pro.json` / `gpt_5mini.json` —— 模型 endpoint 配置
- `models.json` —— 模型注册表

#### [scripts/setup_envs.sh](scripts/setup_envs.sh)
- 见 [LOG_EXEC.md §1.4](LOG_EXEC.md)
- 关键 step：装 uv → 检测 CUDA → 装 sembench venv → 循环装每个 system venv → 用 `import` smoke test 验证

#### [scripts/repeat_experiment.sh](scripts/repeat_experiment.sh)
- 跑同一个 config N 次，结果合并出 std error bar

---

## 6. Estimator / 算法清单

> SemBench **本身不引入新算法**；它是 benchmark。所谓"算法"是各 system 的语义算子实现，列在下表（参考 README "🏆 Adopted By" 章节）。

| Operator | 各 system 的实现 |
|----------|----------------|
| `sem_filter` | LOTUS `df.sem_filter(prompt)` / Palimpzest `pz.Filter` / BigQuery `AI.IF()` / ThalamusDB SQL with semantic predicate |
| `sem_join` | LOTUS `df1.sem_join(df2, join_instruction, cascade_args)` / BigQuery `AI.GENERATE()` 在 ON 子句 |
| `sem_map` | LOTUS `df.sem_map(prompt)` → 写到 `_map` 列 / Palimpzest `pz.Convert` |
| `sem_topk` | LOTUS `df.sem_topk(prompt, K, method='quick', return_stats=True)` |
| `sem_classify` | LOTUS via `sem_map` with classification prompt |

---

## 7. 如何扩展

### 7.1 加新 system（最常见的场景）

**步骤**（与 [LOG_EXEC.md §6.1](LOG_EXEC.md) 对应；这里给"代码视角"的清单）:

1. `requirements/{sys}.txt` —— `-r base.txt` + pin 你的包
2. `src/runner/generic_{sys}_runner/generic_{sys}_runner.py` —— 继承 `GenericRunner`，实现:
   - `get_system_name() -> str`
   - `__init__(...)` 装引擎 + 配置 LM
   - **二选一**: `execute_query(qid)` (单查询) 或 `execute_queries(qids)` (批查询)
   - 在 `execute_query` / `execute_queries` 里调 `_discover_query_impl(qid)` (Code 模式) 或 `_discover_query_text(qid)` (SQL 模式)
   - 填好 `metric.token_usage` / `metric.money_cost`
3. 每个想支持的 scenario，在 `src/scenario/{sc}/runner/{sys}_runner/{sys}_runner.py` 写继承 `Generic{Sys}Runner` 的子类:
   - Code 模式: 实现 `_execute_q{i}()` 方法
   - SQL 模式: 子类几乎空着，只继承；写 `files/{sc}/query/{sys}/Q{i}.sql`
   - Code\* 模式: 子类几乎空着；写 `files/{sc}/query/{sys}/Q{i}.py` 含 `run(data_dir, scale_factor)`
4. 改 [src/run.py:39-47](src/run.py#L39-L47) 加 `"{sys}": "{Sys}Runner"`
5. 在 `src/scenario/{sc}/{sc}_scenario.py` 的 `setup_scenario()` 的 `for system in systems` 里加 `elif system == "{sys}": ...`（即使是 pass，也要显式 elif，否则会抛 `Unsupported system`）

### 7.2 加新 scenario

1. `src/scenario/{name}/` 全套 5 个文件夹建出来（见 §5.3 骨架）
2. `{name}_scenario.py` 实现 5 个方法（见 §4.4）
3. `evaluation/evaluate.py` 继承 `GenericEvaluator`，实现 `_load_domain_data`, `_get_ground_truth`, `_evaluate_q{i}`
4. `preparation/generate_data.py` 实现数据下载 + 采样
5. `setup/bigquery.py` / `setup/flockmtl.py` 实现按 system 把数据 load 到目标 DB
6. 改 [src/run.py:79-87](src/run.py#L79-L87) 加 `"{name}": "{Name}Evaluator"`
7. 改 [src/runner/generic_runner.py:315-340](src/runner/generic_runner.py#L315-L340) 的 `get_scenario_handler` 加 `elif use_case == "{name}": ...`
8. 给每个 system 写 `runner/{sys}_runner/{sys}_runner.py`
9. `files/{name}/query/{system}/Q{i}.*` 写查询文件（gold_sql + natural_language 至少要有）

### 7.3 加新 query

1. `files/{sc}/query/natural_language/Q{N}.txt` —— NL 描述
2. `files/{sc}/query/gold_sql/Q{N}.sql` —— 评测 ground truth 用的 DuckDB SQL
3. 各 system 的实现:
   - Code 模式: 在 `src/scenario/{sc}/runner/{sys}_runner/{sys}_runner.py` 加 `_execute_q{N}()`
   - Code\*: 写 `files/{sc}/query/{sys}/Q{N}.py`
   - SQL: 写 `files/{sc}/query/{sys}/Q{N}.sql`
4. 在 `evaluate.py` 加 `_evaluate_q{N}()`（用 `_generic_retrieval_evaluation` / `_generic_aggregation_evaluation` / `_generic_ranking_evaluation` 之一 或自写）
5. 更新 `files/{sc}/query/coverage.json`

### 7.4 加新 metric

- 在 [src/evaluator/generic_evaluator.py:30-90](src/evaluator/generic_evaluator.py#L30-L90) 加 `@dataclass class QueryMetric{X}`
- 加 `_generic_{x}_evaluation(self, sys_df, gt_df) -> QueryMetric{X}` helper
- 在 `_evaluate_q{i}` 里返回它

### 7.5 加新 model（pricing）

只需要在两处加条目:
- [src/runner/generic_lotus_runner/generic_lotus_runner.py:24-54](src/runner/generic_lotus_runner/generic_lotus_runner.py#L24-L54) 的 `PRICING` dict
- [src/runner/generic_bigquery_runner/generic_bigquery_runner.py:110-131](src/runner/generic_bigquery_runner/generic_bigquery_runner.py#L110-L131) 的 `MODEL_PRICES` dict
- ThalamusDB 也有自己的 PRICING（见 [src/runner/generic_thalamusdb_runner/generic_thalamusdb_runner.py:177-208](src/runner/generic_thalamusdb_runner/generic_thalamusdb_runner.py#L177-L208)）

---

## 8. 文件命名约定（路径模板）

| 路径模板 | 内容 |
|---------|------|
| `files/{sc}/data/sf_{N}/*.csv` | 该 scale_factor 的数据（CSV + 图像 / 音频文件） |
| `files/{sc}/data/sf_{N}/{table}.csv.hash` | 该数据文件的内容哈希（用于检测改动） |
| `files/{sc}/query/natural_language/Q{i}.txt` | 自然语言查询描述（人读 + 部分 system 直接用） |
| `files/{sc}/query/gold_sql/Q{i}.sql` | 评测用的 DuckDB SQL，跑出来就是 ground truth |
| `files/{sc}/query/{system}/Q{i}.sql` | SQL 模式查询（BigQuery / FlockMTL） |
| `files/{sc}/query/{system}/Q{i}.py` | Code\* 模式查询（必须有 `run(data_dir, scale_factor)`） |
| `files/{sc}/query/coverage.json` | 该 scenario 各 system 已实现的 query 编号 |
| `files/{sc}/raw_results/{system}/Q{i}.csv` | 每次跑 runner 落的原始结果 |
| `files/{sc}/raw_results/ground_truth/Q{i}.csv` | evaluator 跑 gold_sql 落的 ground truth |
| `files/{sc}/metrics/{system}.json` | 一个 system × scenario 的全部 query 指标（runner + evaluator 联合写） |
| `figures/{sc}/*.png` | `plot.py` 输出 |
| `config/system/{sys}/*.json` | 各 system 的运行配置（model / policy / ...） |
| `.venvs/{sys}/` | uv 装的隔离 venv |

---

## 9. 参考资料

### 内部
- [README.md](README.md) —— scenario 矩阵 + 文档门面
- [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) —— 多 venv 的 why + how
- [docs/submit.html](docs/submit.html) `#implementation-guide` —— 官方扩展指南
- [LOG.md](LOG.md) —— 修改历史 + pitfall
- [LOG_EXEC.md](LOG_EXEC.md) —— 可复制粘贴的命令

### 外部
- Paper: https://arxiv.org/abs/2511.01716
- GitHub upstream: https://github.com/SemBench/SemBench
- Online Leaderboard: https://sembench.org
- Multi-modal datasets (Google Drive): https://drive.google.com/drive/folders/1pqf8DKFai16MR80Z7pcls5FgBbom-IJt
- uv: https://docs.astral.sh/uv/
- LOTUS: https://lotus-data.github.io / 本地副本 [Desktop/AllSQPE/LOTUS/](../AllSQPE/LOTUS/)
- Palimpzest: 本地副本 [Desktop/AllSQPE/Palimpzest/](../AllSQPE/Palimpzest/)
- ThalamusDB: 本地副本 [Desktop/AllSQPE/ThalamusDB/](../AllSQPE/ThalamusDB/)
- CAESURA: 本地副本 [Desktop/AllSQPE/CAESURA/](../AllSQPE/CAESURA/)
- FlockMTL: 本地副本 [Desktop/AllSQPE/FlockMTL/](../AllSQPE/FlockMTL/)
- Sema / Unify: 本地副本 [Desktop/AllSQPE/Sema/](../AllSQPE/Sema/) / [Desktop/AllSQPE/Unify/](../AllSQPE/Unify/)

---

## 10. L0 / L1 / L2 标注层级（占位）

> **当前 SemBench 不是"paper-derived 单方法 repo"**（不像 Naru / MSCN / CoLSE 那种从原始 paper 仓库逐步派生到 lecarb 的层叠），所以**不存在 L0 / L1 / L2 三层对照表**。
>
> 如果未来需要做 [[Workplan/CLAUDE-md]] §5.5 教学注释 (annotation pass)，本章是占位 —— 届时按"L0 = upstream 当前 HEAD"单层处理即可，无需 ADD / MODIFY 表格。
>
> 改 SemBench 自带代码做注释时，仍然受 [Desktop/CLAUDE.md §5.5 D](../CLAUDE.md) 硬约束限制（只增加 comment / docstring，不动 statement，不改行号，逐文件 ast.parse 验证）。
