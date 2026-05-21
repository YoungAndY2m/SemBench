"""
============================================================================
教学注释 (Annotation Pass) — Cars scenario 的 ThalamusDB 数据初始化
============================================================================

ThalamusDB = 基于 DuckDB + FlockMTL extension 的语义查询引擎. SemBench 用它跑
cars scenario 的实验. 本文件做 3 件事:

1. **__init__**: 打开 DuckDB 文件 `{db_folder}/{db_name}.duckdb` 作为
   ThalamusDB 实例; 安装 + 加载 FlockMTL extension (语义函数); 用 OpenAI API key
   建一个 SECRET 让 FlockMTL 能调 LLM; 注册 model_name 到 ThalamusDB 的模型表.

2. **_upload_file_to_db**: 用 read_csv_auto 把 CSV 文件加载成 DuckDB 表
   (CREATE OR REPLACE TABLE).

3. **setup_data**: 加载 4 个 cars 数据集 (主表 + audio + complaints + images)
   到 4 个 DuckDB 表 (cars / car_audio / car_complaints / car_images).

**DuckDB extension API 速记**:
- `install_extension(name, repository="community")` = 从 community repo 装扩展
- `load_extension(name)` = 加载已装的扩展到当前 connection
- FlockMTL extension 提供 LLM 相关 SQL 函数 (llm_complete, llm_filter 等)

**SECRET 是 DuckDB 的密钥管理**: `CREATE SECRET (TYPE OPENAI, API_KEY '...')`
存 OpenAI 密钥, FlockMTL 内部用 secret 而不是直接读环境变量, 避免密钥泄露到
SQL plan / log 里.

**CARS_FILES_DIR**: 用 pathlib 推到当前文件 4 级父目录上的 `files/cars/data/`
路径; 用 `.resolve()` 走绝对路径避免相对路径歧义.

注释说明: 本注释 pass 只增加 comment, 不改任何原始代码.
============================================================================
"""


import os
from pathlib import Path

import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

# 用 pathlib 推到当前文件 4 级父目录上的 files/cars/data/ 路径; .resolve() 走
# 绝对路径避免相对路径歧义. 这是 cars scenario 的 default DuckDB 文件落盘位置.
CARS_FILES_DIR = os.path.abspath(
    Path(__file__).resolve().parents[4] / "files" / "cars" / "data"
)


# ============================================================================
# ThalamusDBCarsSetup — 给 cars scenario 创建 ThalamusDB (DuckDB + FlockMTL) 环境
# ============================================================================
class ThalamusDBCarsSetup:
    def __init__(self, model_name: str = "gpt-4o-mini", db_name:str = 'cars_database', load_extensions: bool = True, db_folder: str = CARS_FILES_DIR):
        """
        Initializes the ThalamusDB connection using environment variables.
        """
        # 必须先在环境里设 OPENAI_API_KEY (用 .env / export); 否则 FlockMTL 没法
        # 调 LLM. 这里 fail-fast 抛 ValueError, 比后面 SQL 运行时报错好定位.
        if os.environ.get('OPENAI_API_KEY') is None:
            raise ValueError("Environment variable OPENAI_API_KEY is not set.")

        # duckdb.connect(path) 打开/创建一个 DuckDB 文件; 之后所有 SQL 都走它.
        self.thalamusdb_conn = duckdb.connect(os.path.join(db_folder, f"{db_name}.duckdb"))

        if load_extensions:
            # FlockMTL 是 SQPE engine, paper VLDB 2026 (见 AllSQPE/FlockMTL/);
            # 装到 DuckDB 后才有 llm_complete / llm_filter 等语义函数.
            self.thalamusdb_conn.install_extension("flockmtl", repository="community")
            self.thalamusdb_conn.load_extension("flockmtl")

            # CREATE SECRET = DuckDB 的密钥管理机制, 内部存 OpenAI API key, 后续
            # FlockMTL 调 LLM 时引用 secret (不暴露到 SQL plan / log).
            self.thalamusdb_conn.execute(
                f"""CREATE SECRET (TYPE OPENAI,API_KEY '{os.environ.get('OPENAI_API_KEY')}');"""
            )

            # 检查 model 是否已注册到 ThalamusDB; 没有就 CREATE MODEL 注册.
            # GET MODELS 是 FlockMTL 的元 SQL, 返回所有已注册模型清单.
            # CREATE MODEL 参数: (alias_in_SQL, openai_model_name, provider, config_json)
            #   - tuple_format='json': LLM 输出按 JSON 解析
            #   - batch_size=32: 32 行一批喂 LLM (节省调用次数 + 提速)
            #   - temperature=0.7: 不 0 也不 1 的中间值, 兼顾稳定性 + 多样性
            # .replace("model_name", model_name) 是脏 trick 把字符串里 4 处
            # "model_name" 占位符全替换 (跟 jinja 同思想但 100x 简陋).
            if not model_name in self.thalamusdb_conn.execute("GET MODELS;").fetchdf()["model"].tolist():
                self.thalamusdb_conn.execute("""
                    CREATE MODEL(
                    'model_name',
                    'model_name',
                    'openai',
                    {"tuple_format": "json", "batch_size": 32, "model_parameters": {"temperature": 0.7}}
                    );
                """.replace("model_name", model_name))


    def _upload_file_to_db(self, csv_path: str, table_name: str):
        # 把 CSV 直接 load 进 DuckDB 表; read_csv_auto 自动 sniff 列类型.
        # CREATE OR REPLACE TABLE = 存在就 drop 重建, 不在就建 (幂等).
        # ★ 注意: f-string 注入 csv_path 没 escape, 如果 path 含单引号会注入失败;
        # SemBench 自己控制 path 所以 OK, 但搬到 user-input 场景要 fix.
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found at path: {csv_path}. Please run the download script first.")

        self.thalamusdb_conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path}');
            """)


    def setup_data(self, data_dir: str, scale_factor: int = 157376):
        # scale_factor=157376 是 cars 数据集行数 (paper 默认 SF); 不同 SF 对应不同
        # 子目录 sf_<N>/, 内含 4 个 CSV (主表/audio/complaints/images).
        sf_dir = os.path.join(data_dir, "data", f"sf_{scale_factor}")

        self._upload_file_to_db(
            csv_path=os.path.join(sf_dir, f"car_data_{scale_factor}.csv"),
            table_name="cars"
        )

        self._upload_file_to_db(
            csv_path=os.path.join(sf_dir, f"audio_car_data_{scale_factor}.csv"),
            table_name="car_audio"
        )

        self._upload_file_to_db(
            csv_path=os.path.join(sf_dir, f"text_complaints_data_{scale_factor}.csv"),
            table_name="car_complaints"
        )

        self._upload_file_to_db(
            csv_path=os.path.join(sf_dir, f"image_car_data_{scale_factor}.csv"),
            table_name="car_images"
        )

    def get_connection(self):
        """
        Returns the ThalamusDB connection.
        """
        return self.thalamusdb_conn

    def run_query(self, query: str):
        self.thalamusdb_conn.execute(query)
