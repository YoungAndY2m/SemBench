# SemBench — 执行命令手册（LOG_EXEC）

> 此文件汇总所有"实际执行"用的命令：环境准备 → 数据 → 实验跑测 → 进度查看 → 结果分析。
> 配合 [LOG.md](LOG.md)（"为什么/改了什么"）与 [LOG_STRUCTURE.md](LOG_STRUCTURE.md)（"代码长啥样"）一起读。
> 由于 SemBench 当前还未列入正式 PLAN，本文件以 README + ENVIRONMENT_SETUP + submit.html `#implementation-guide` 为权威来源。

---

## 0. 前置：每次新终端必做

```bash
cd ~/Desktop/SemBench

# 激活 orchestrator venv（跑 run.py / 评测 / 画图都在这里面）
source .venvs/sembench/bin/activate

# 加载 API key 等（若 .env 存在）
# python-dotenv 会自动 load，但手工 export 也可以
set -a; [ -f .env ] && . .env; set +a
```

---

## 1. 从零环境配置（Fresh clone 后第一次）

### 1.1 系统依赖（一次性 sudo —— 一般 Ubuntu 都已具备）

```bash
sudo apt update
sudo apt install -y curl build-essential pkg-config \
    libsm6 libxext6 libxrender-dev \
    tesseract-ocr libtesseract-dev \
    poppler-utils    # pdf2image 需要
```

### 1.2 用户态工具链（一次性，无 sudo）

```bash
# uv —— 多 venv 包管理器。脚本会自动装，但手动也行：
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version    # 验证
```

### 1.3 Clone（已完成；保留命令供 reference）

```bash
# 当前仓库已 clone 在 ~/Desktop/SemBench
# upstream: https://github.com/SemBench/SemBench.git
# origin:   https://github.com/YoungAndY2m/SemBench.git
git remote -v
git log -1 --oneline    # 确认 HEAD
```

### 1.4 一键创建所有 venv

```bash
cd ~/Desktop/SemBench

# 全装（sembench + 6 个系统 venv，每个 1-3 分钟）
bash scripts/setup_envs.sh

# 或者只装少数系统（推荐，节省时间）
bash scripts/setup_envs.sh lotus

# 多个
bash scripts/setup_envs.sh lotus palimpzest

# 列出可装的系统
bash scripts/setup_envs.sh --list
```

脚本会自动:
1. 装 uv（若没有）
2. 检测 `nvidia-smi`，相应装 CUDA 或 CPU 版 PyTorch
3. 在 `.venvs/` 下建 `sembench/`、`lotus/`、`palimpzest/` 等
4. 装完后用 `python -c "import lotus"` 这类语句做 smoke test

### 1.5 配置 `.env`

```bash
cp .env.example .env
# 编辑：
# OPENAI_API_KEY=...        （LOTUS / Palimpzest / CAESURA 用 OpenAI 时需要）
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json
# GCLOUD_PROJECT=your-gcp-project
# KAGGLE_USERNAME=... KAGGLE_KEY=...   （部分 scenario 数据集走 Kaggle）
```

### 1.6 数据：自动下载或重新生成

> 仓库**自带**了 `files/{scenario}/data/sf_*/` 的 demo 数据，但 README 明确说"演示用，做实验前请删除并重新生成"。

```bash
# 删除 demo 数据（如果要做正式实验）
rm -rf files/movie/data files/animals/data files/cars/data \
       files/ecomm/data files/mmqa/data files/medical/data

# 不需要手动跑 download —— 第一次跑 run.py 时，scenario_handler.setup_scenario()
# 会自动从 Google Drive 下载并构造数据库（见 src/scenario/{scenario}/preparation/generate_data.py）
```

数据下载源（参考）: https://drive.google.com/drive/folders/1pqf8DKFai16MR80Z7pcls5FgBbom-IJt

### 1.7 烟测：单 query 跑通

```bash
source .venvs/sembench/bin/activate

# 最小可运行 case：LOTUS + Movie + Q1 + scale 2000
python3 src/run.py \
    --systems lotus \
    --use-cases movie \
    --queries 1 \
    --model gemini-2.5-flash \
    --scale-factor 2000
```

预期看到:
- `Per-system venvs detected: lotus, palimpzest, ...`（如果都装了）
- `[isolated] Using venv: .../.venvs/lotus/bin/python` —— 走的隔离子进程
- `Q1: ✅ {time}s, {n} rows, {tokens} tokens, ${cost}`
- 结果写到:
  - `files/movie/raw_results/lotus/Q1.csv`
  - `files/movie/metrics/lotus.json`

---

## 2. 实验执行

### 2.1 准备日志目录

```bash
mkdir -p logs/$(date +%Y%m%d)
```

### 2.2 单系统 × 单 scenario × 全 queries

```bash
python3 src/run.py \
    --systems lotus \
    --use-cases movie \
    --model gemini-2.5-flash \
    --scale-factor 2000 \
  2>&1 | tee logs/$(date +%Y%m%d)/lotus_movie_full.log
```

### 2.3 多系统并跑（每个系统跑在各自隔离 venv 中）

```bash
python3 src/run.py \
    --systems lotus palimpzest thalamusdb bigquery \
    --use-cases movie \
    --queries 1 5 \
    --model gemini-2.5-flash \
    --scale-factor 2000 \
  2>&1 | tee logs/$(date +%Y%m%d)/all_systems_movie_Q1Q5.log
```

### 2.4 多 scenario 比较

```bash
python3 src/run.py \
    --systems lotus \
    --use-cases movie animals cars \
    --queries 1 \
    --model gemini-2.5-flash \
    --scale-factor 2000
```

### 2.5 重复实验拿 error bar

> 由 [scripts/repeat_experiment.sh](scripts/repeat_experiment.sh) 驱动；先 cat 看里面的参数，按需改。

```bash
cd scripts
cat repeat_experiment.sh   # 看一下默认参数
./repeat_experiment.sh     # 通常跑 5 次
cd ..
```

### 2.6 跳过 setup（数据已就绪时加速）

```bash
python3 src/run.py --systems lotus --use-cases movie --queries 1 \
    --scale-factor 2000 --skip-setup
```

### 2.7 关闭 venv 隔离（debug 用，所有系统跑在 sembench venv）

```bash
python3 src/run.py --systems lotus --use-cases movie --queries 1 \
    --scale-factor 2000 --no-isolation
```

### 2.8 Query ID 两种格式都支持

```bash
python3 src/run.py --queries 1 5 10            # 整数
python3 src/run.py --queries Q1 Q5 Q10         # Q-prefix —— 内部 strip "Q" 转 int
```

### 2.9 切换模型

支持的模型（[src/runner/generic_lotus_runner/generic_lotus_runner.py:24-54](src/runner/generic_lotus_runner/generic_lotus_runner.py#L24-L54) 的 `PRICING` 表）:

- `gemini-2.5-flash`（**默认**）
- `gemini-2.5-flash-lite`
- `gemini-2.5-pro`
- `gemini-2.0-flash`
- `gpt-5`
- `gpt-5-mini`
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4o-audio-preview`

未在 PRICING 表里的模型会跑通但 cost 字段为 0。

```bash
python3 src/run.py --systems lotus --use-cases movie --queries 1 \
    --model gpt-4o-mini --scale-factor 2000
```

### 2.10 scale-factor 的取值范围

> 不同 scenario 含义不同（参考 [LOG.md](LOG.md) 已知 pitfall #5）

| Scenario | 默认 / 推荐 | 含义 |
|----------|------------|------|
| movie    | `2000`     | 采样 review 行数 |
| cars     | `157376`   | complaints 全集大小（默认在 [files/cars/query/lotus/Q1.py:4](files/cars/query/lotus/Q1.py#L4)） |
| animals / ecomm / mmqa / medical | 未确认 | 看对应 `src/scenario/{name}/preparation/generate_data.py` |

```bash
# 跑前先看一眼那个 scenario 的 preparation 脚本
ls src/scenario/<scenario>/preparation/
```

---

## 3. 评测与可视化

### 3.1 跑评测（已内嵌在 run.py 末尾，单独执行 = 重跑评测）

```bash
# run.py 跑完会自动调 evaluator —— 见 src/run.py:321-343
# 单独重跑某个 evaluator（python 交互式）：
python3 -c "
import sys; sys.path.insert(0, 'src')
from run import get_evaluator
ev = get_evaluator('movie')('movie', 2000)
ev.evaluate_system('lotus', queries=[1, 5])
"
```

### 3.2 生成可视化

```bash
python3 src/plot.py
# 输出: figures/{scenario}/*.png
```

### 3.3 生成 LaTeX 表（论文用）

```bash
python3 src/table_brick_design_avg.py
# 或者旧版：
python3 src/table_brick_design.py
```

### 3.4 生成分析报告

```bash
python3 scripts/analysis.py
# 输出: analysis_results/
```

---

## 4. 进度查看与 debug

SemBench 没有内置 progress bar 工具，靠 stdout + log 文件。常用：

```bash
# 实时跟踪运行日志
tail -f logs/$(date +%Y%m%d)/lotus_movie_full.log

# 一行一个 query 提取耗时
grep -E "Q\d+:" logs/.../*.log

# 看资源占用
htop -p $(pgrep -f run.py)

# 看 LLM 调用是否还在跑（litellm 默认会打 INFO 行）
grep -i "litellm\|openai\|gemini" logs/.../*.log | tail -20

# 看是否触发了 BigQuery 5s 等待
grep -i "inference logs to materialize" logs/.../*.log

# 强杀 hang 住的 LOTUS 进程（run.py 自带 os._exit(0) 一般不需要）
pgrep -f "run_worker.py" | xargs -r kill -9
```

### 4.1 看本次结果

```bash
# 最近一次跑出的 metrics（系统 × scenario 行）
cat files/movie/metrics/lotus.json | python3 -m json.tool

# 最近一次跑出的原始结果（每 query 一份 CSV）
ls files/movie/raw_results/lotus/
head files/movie/raw_results/lotus/Q1.csv

# ground truth 也在这下面（DuckDB 跑 gold SQL 生成）
ls files/movie/raw_results/ground_truth/
```

---

## 5. 常见错误速查

| 报错 / 现象 | 原因 | 解法 |
|------------|------|------|
| `ModuleNotFoundError: No module named 'lotus'` | 在 `sembench` venv 里跑了直接 import；没走隔离 | 让 `run.py` 走 `--isolation`（默认）；或装到当前 venv |
| `No venv found for {system} at .venvs/{system}` | `setup_envs.sh` 没装该系统 | `bash scripts/setup_envs.sh {system}` |
| `numpy` 相关 ABI 报错 | 把 lotus + palimpzest 装到同一 venv | 用 `setup_envs.sh` 的多 venv 模式，不要用根目录 `requirements.txt` |
| `Cost will exclude this model` warning | 模型不在 [PRICING](src/runner/generic_lotus_runner/generic_lotus_runner.py#L24-L54) 表里 | 在表里加一条；或忽略 cost 字段 |
| BigQuery 跑完后停 5-15s | 等 `inference_logs` 表 materialize（[generic_bigquery_runner.py:186-282](src/runner/generic_bigquery_runner/generic_bigquery_runner.py#L186-L282)） | 正常现象，不是 hang |
| `usage not materialized yet (no rows)` | 同上，inference logs 还没写完 | 内置 3 次重试，等就行 |
| `FileNotFoundError: ... gold_sql/Q*.sql` | 评测时找不到 ground truth SQL | 检查 [files/{scenario}/query/gold_sql/](files/movie/query/gold_sql/) 是否完整 |
| `Unknown use case: detective` | 引用了已废弃的 use case（见 [LOG.md](LOG.md) #3） | 改用 `movie / animals / cars / ecomm / mmqa / medical` |
| Q1.csv 是空的 | 语义算子（如 `sem_filter`）返回空 —— 通常是 prompt 没匹配上数据 | 看 prompt 是否正确；看 input 数据是否真的非空 |
| `__WORKER_RESULT__` 解析失败 | worker stdout 包含了这个 marker，破坏正则 | 改 worker 的 print，不要让 user-data 经过 stdout |

---

## 6. 接入自己系统 / scenario 的最小步骤

> 摘自 submit.html `#implementation-guide` 第 4 步 + [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) 末尾的 "Adding a New System"

### 6.1 加新系统（以 `myengine` 为例）

```bash
# (1) 写依赖
cat > requirements/myengine.txt <<'EOF'
-r base.txt
myengine-package==1.0.0
EOF

# (2) 装隔离 venv
bash scripts/setup_envs.sh myengine

# (3) 写 base runner —— 见 LOG_STRUCTURE.md §5
mkdir -p src/runner/generic_myengine_runner
# 内容参考 src/runner/generic_lotus_runner/generic_lotus_runner.py

# (4) 改 src/run.py 的 get_runner_class 字典
#     在 runner_classes 里加: "myengine": "MyEngineRunner",

# (5) 给每个想支持的 scenario 写 per-scenario runner
mkdir -p src/scenario/movie/runner/myengine_runner
# 实现 _execute_q1(self) -> pd.DataFrame 等

# (6) 跑测试
python3 src/run.py --systems myengine --use-cases movie --queries 1
```

### 6.2 加新 scenario

```bash
# (1) 建目录骨架
mkdir -p src/scenario/myscene/{evaluation,preparation,runner,setup}

# (2) 写 scenario handler （类比 src/scenario/movie/movie_scenario.py）
# (3) 写 evaluator （类比 src/scenario/movie/evaluation/evaluate.py）
# (4) 改 src/run.py 的 get_evaluator 字典 + src/runner/generic_runner.py:310 的 get_scenario_handler
# (5) 把 gold_sql + natural_language + 各 system 的 query 文件放进 files/myscene/query/
# (6) 实现各 system 的 per-scenario runner
```

---

## 7. 与 Workplan 的映射

> 当前 [[Workplan/week-3]] 内的 PLAN.md 未将 SemBench 列入主线；本仓库目前作为 reconnaissance。
> 一旦决定迁移 ARELY，应：
> 1. 在 [[Workplan/week-4]] 新建 PLAN entry，描述"在 SemBench 上做 sem_filter / sem_join selectivity 估计"的 step
> 2. 把本 LOG_EXEC.md 的对应章节按 PLAN step 重排

| LOG_EXEC 章节 | PLAN.md 对应 step | 说明 |
|--------------|------------------|------|
| §1.1-1.6 | （未来 week-4 Step 0） | 一次性环境配置 |
| §1.7 | （未来 week-4 Step 1） | 烟测 |
| §2.1-2.5 | （未来 week-4 Step 2-4） | 跑基线 baseline 拿原始 cost / latency / quality |
| §2.6-2.10 | （未来 week-4 Step 5+） | 参数 sweep |
| §3 | （未来 week-4 Step Final） | 可视化 + 表格 |

---

## 8. 速查命令汇总卡（一屏看完）

```bash
# 环境
cd ~/Desktop/SemBench
source .venvs/sembench/bin/activate

# 装系统
bash scripts/setup_envs.sh lotus palimpzest

# 跑一个 query
python3 src/run.py --systems lotus --use-cases movie --queries 1 \
    --model gemini-2.5-flash --scale-factor 2000

# 跑全部 query
python3 src/run.py --systems lotus --use-cases movie \
    --model gemini-2.5-flash --scale-factor 2000

# 多系统比较
python3 src/run.py --systems lotus bigquery thalamusdb \
    --use-cases movie --queries 1 5 \
    --model gemini-2.5-flash --scale-factor 2000

# 出图 + 表
python3 src/plot.py
python3 src/table_brick_design_avg.py

# 看结果
cat files/movie/metrics/lotus.json | python3 -m json.tool
ls files/movie/raw_results/lotus/

# 数据重置
rm -rf files/movie/data/

# 重复实验（编辑参数后）
bash scripts/repeat_experiment.sh
```
