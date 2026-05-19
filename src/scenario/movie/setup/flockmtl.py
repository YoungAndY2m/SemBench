"""
============================================================================
SemBench L1 wrapper — movie/setup/flockmtl.py
============================================================================
教学注释 pass (L1 ADD) by Claude.

这个文件干什么 (一句话):
  FlockMTLMovieSetup 类 — 把 "建 DuckDB 文件 + 装 flockmtl extension + 配 OpenAI secret +
  CREATE MODEL + 加载 movie CSV" 一套 boilerplate 包成单 class. 由 FlockMTLRunner.__init__ 调用.

----------------------------------------------------------------------------
关键步骤 (按 __init__ 顺序):
----------------------------------------------------------------------------
1. 验证 OPENAI_API_KEY env var (硬编码 OpenAI; 不支持其它 provider)
2. duckdb.connect(): 打开 *持久* DuckDB 文件 movie_database.duckdb (复用)
3. install_extension("flockmtl", repository="community"): 从 DuckDB community catalog 装
4. load_extension("flockmtl"): 加载到当前 conn (触发 L0 flock_extension.cpp:LoadInternal)
5. CREATE SECRET: 把 API key 注入 DuckDB SecretManager (见 L0 secret_manager.cpp)
6. GET MODELS: 检查 model_name 是否已注册; 没有 → CREATE MODEL DDL

----------------------------------------------------------------------------
⚠ 关键缺陷 (LOG.md "遗留问题"):
----------------------------------------------------------------------------
- 硬编码 OpenAI — SemBench 默认 --model gemini-2.5-flash 但本 setup *只接受 OPENAI_API_KEY*
- INSTALL flockmtl 用旧名字 — L0 已改为 "flock" (FlockExtension::Name() = "flock"); community
  catalog 仍叫 flockmtl 但与 L0 binary name 对不上 (LOG_STRUCTURE.md §10.3 #5)
- CREATE MODEL 用字符串 .replace("model_name", model_name) — 不是 SQL parameter binding,
  存在 SQL injection 风险 (实际中 model_name 来自 CLI, 不算严重)

----------------------------------------------------------------------------
零基础读者预备知识:
----------------------------------------------------------------------------
- `Path(__file__).resolve().parents[4]`:
    pathlib API; `__file__` 是当前文件路径, .resolve() 转绝对路径, .parents[N] 跳 N 级父目录.
    parents[4] = 跳 4 级 → SemBench/ 根目录.

- `duckdb.connect(path)`:
    DuckDB Python API; 打开 *持久 db file* (vs `:memory:` 内存). 多次 connect 同一 path 共享数据.

- `install_extension(name, repository="community")`:
    DuckDB 从 community catalog 下载 .duckdb_extension binary 装到本地缓存.
    第一次跑慢 (下载), 后续从本地加载.

- f-string + triple-quoted string:
    多行字符串里嵌入变量插值 (本文件 line 29 / 47 用法).
    注意 SQL 用 单引号 包成字符串值 — 比普通 .format() 简洁.
============================================================================
"""

import os
from pathlib import Path

import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

# parents[4] 跳 4 级: setup/ → movie/ → scenario/ → src/ → SemBench/
# + "files/movie/data" → SemBench/files/movie/data/ (持久 db + CSV 都放这里)
MOVIE_FILES_DIR = os.path.abspath(
    Path(__file__).resolve().parents[4] / "files" / "movie" / "data"
)

class FlockMTLMovieSetup:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initializes the FlockMTL connection using environment variables.
        """
        # 1. 验证 API key (硬约束: 必须 OPENAI_API_KEY env var 存在)
        if os.environ.get('OPENAI_API_KEY') is None:
            raise ValueError("Environment variable OPENAI_API_KEY is not set.")

        # 2. 打开持久 DuckDB 文件 movie_database.duckdb (后续 query 直接读取)
        self.flockmtl_conn = duckdb.connect(os.path.join(MOVIE_FILES_DIR, "movie_database.duckdb"))
        

        # 3. 装 + 加载 flockmtl extension (community catalog binary)
        # 第一次跑会下载, 后续从本地缓存 ~/.duckdb/extensions/ 加载
        self.flockmtl_conn.install_extension("flockmtl", repository="community")
        self.flockmtl_conn.load_extension("flockmtl")

        # 4. 创建 OpenAI secret (用户 SQL 内不必再传 api_key, FlockMTL 自动从 SecretManager 取)
        # ⚠ 字符串拼接 API key — SQL injection 风险 (实际中 env var 受信任)
        self.flockmtl_conn.execute(
            f"""CREATE SECRET (TYPE OPENAI,API_KEY '{os.environ.get('OPENAI_API_KEY')}');"""
        )

        # 5. 检查 model 是否已注册; 没有 → CREATE MODEL DDL (FlockMTL 自定义 DDL, L0 query_parser 解析)
        # GET MODELS; 返回 DuckDB DataFrame, .fetchdf()["model"] 取 model name 列
        if not model_name in self.flockmtl_conn.execute("GET MODELS;").fetchdf()["model"].tolist():
            # CREATE MODEL DDL 模板:
            # 'model_name' (alias for SQL) = 第 1 arg
            # 'model_name' (actual API model) = 第 2 arg
            # 'openai' = provider
            # 配置: tuple_format=json (传 LLM 的数据格式), batch_size=32, temperature=0.7
            # ⚠ .replace("model_name", model_name) 同时替换两个 'model_name' 字面 — 简单粗暴
            self.flockmtl_conn.execute("""
                CREATE MODEL(
                'model_name',
                'model_name', 
                'openai', 
                {"tuple_format": "json", "batch_size": 32, "model_parameters": {"temperature": 0.7}}
                );
            """.replace("model_name", model_name))


    def _upload_file_to_db(self, csv_path: str, table_name: str):
        # 用 DuckDB CREATE TABLE FROM read_csv_auto — DuckDB 自动推断 schema
        # CREATE OR REPLACE: 重跑 setup 时不报已存在错
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found at path: {csv_path}. Please run the download script first.")

        self.flockmtl_conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path}');
            """)
        


    def setup_data(self, data_dir: str):
        # 加载 movie scenario 的 2 个 CSV → 2 个 DuckDB 表
        # 表名带 _2000 后缀 (与 scale_factor=2000 关联) — SQL 模板 (Q1.sql 等) 内引用这俩表
        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "data/Movies_2000.csv"), 
            table_name="movies_2000"
        )

        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "data/Reviews_2000.csv"), 
            table_name="reviews_2000"
        )


    def get_connection(self):
        """
        Returns the FlockMTL connection.
        """
        # Runner (FlockMTLRunner) 用此 getter 拿 conn 后赋给 self.flockmtl_conn
        # 让 GenericFlockMTLRunner.execute_queries 内 self.flockmtl_conn.execute() 能跑
        return self.flockmtl_conn
