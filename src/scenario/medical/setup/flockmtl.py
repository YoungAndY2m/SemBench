
"""
============================================================================
SemBench L1 wrapper — medical/setup/flockmtl.py
============================================================================
教学注释 pass (L1 ADD) by Claude.

与 [movie/setup/flockmtl.py] 高度相似 (复制 + 修改名字 + 加表), 关键差别:
  - 5 个表 (vs movie 的 2 个): patients / lung_audio / symptoms_texts / x_ray_images / skin_images
  - __init__ 多 3 参数: db_name (允许多 db 共存), load_extensions (allow skip — multi-scenario reuse),
    db_folder (可指定其它路径; mmqa 复用本 setup 时会传不同 folder)
  - setup_data 接 scale_factor — 文件名变 (patient_data.csv vs patient_data_<sf>.csv)
  - run_query 额外方法 — 让 caller 跑 SQL 而不必先 get_connection (boilerplate)

⚠ 注意: 本 setup *能用* 但 runner 是 **空壳** (没调 setup_data):
  见 [medical/runner/flockmtl_runner/flockmtl_runner.py] — 仅 __init__ 设 conn, 没载数据
  导致 medical scenario 跑时表不存在; raw_results 仅 Q1 + Q10 — 推测是手动跑了 2 个 query 留下的产物
============================================================================
"""

import os
from pathlib import Path

import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

MEDICAL_FILES_DIR = os.path.abspath(
    Path(__file__).resolve().parents[4] / "files" / "medical" / "data"
)


class FlockMTLMedicalSetup:
    # 比 movie 版多 3 参数: db_name / load_extensions / db_folder — 让本 class 可被 mmqa setup 复用
    def __init__(self, model_name: str = "gpt-4o-mini", db_name:str = 'medical_database', load_extensions: bool = True, db_folder: str = MEDICAL_FILES_DIR):
        """
        Initializes the FlockMTL connection using environment variables.
        """
        if os.environ.get('OPENAI_API_KEY') is None:
            raise ValueError("Environment variable OPENAI_API_KEY is not set.")

        self.flockmtl_conn = duckdb.connect(os.path.join(db_folder, f"{db_name}.duckdb"))

        if load_extensions:
            self.flockmtl_conn.install_extension("flockmtl", repository="community")
            self.flockmtl_conn.load_extension("flockmtl")

            self.flockmtl_conn.execute(
                f"""CREATE SECRET (TYPE OPENAI,API_KEY '{os.environ.get('OPENAI_API_KEY')}');"""
            )

            if not model_name in self.flockmtl_conn.execute("GET MODELS;").fetchdf()["model"].tolist():
                self.flockmtl_conn.execute("""
                    CREATE MODEL(
                    'model_name',
                    'model_name', 
                    'openai', 
                    {"tuple_format": "json", "batch_size": 32, "model_parameters": {"temperature": 0.7}}
                    );
                """.replace("model_name", model_name))


    def _upload_file_to_db(self, csv_path: str, table_name: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found at path: {csv_path}. Please run the download script first.")

        self.flockmtl_conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path}');
            """)


    # setup_data: 5 个表的 CSV 加载
    # scale_factor 默认 11112 (medical 默认 scale = 11112); 不同 SF 用不同 csv 文件名后缀
    def setup_data(self, data_dir: str, scale_factor: int = 11112):
        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "data/patient_data.csv" if scale_factor == 11112 else f"data/patient_data_{scale_factor}.csv"), 
            table_name="patients"
        )

        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "data/audio_lung_data.csv" if scale_factor == 11112 else f"data/audio_lung_data_{scale_factor}.csv"), 
            table_name="lung_audio"
        )

        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "data/text_symptoms_data.csv" if scale_factor == 11112 else f"data/text_symptoms_data_{scale_factor}.csv"), 
            table_name="symptoms_texts"
        )

        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "data/image_x_ray_data.csv" if scale_factor == 11112 else f"data/image_x_ray_data_{scale_factor}.csv"), 
            table_name="x_ray_images"
        )

        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "data/image_skin_data.csv" if scale_factor == 11112 else f"data/image_skin_data_{scale_factor}.csv"),
            table_name="skin_images"
        )

    def get_connection(self):
        """
        Returns the FlockMTL connection.
        """
        return self.flockmtl_conn
    
    # run_query: boilerplate helper — 让 caller 不必 get_connection 后再 execute
    # 在 medical/mmqa 内部某些 manual setup script 用得到 (e.g. setup_db.sql 手动跑 DDL)
    def run_query(self, query: str):
        self.flockmtl_conn.execute(query)
