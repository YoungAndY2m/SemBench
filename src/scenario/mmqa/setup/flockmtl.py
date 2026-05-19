"""
============================================================================
SemBench L1 wrapper — mmqa/setup/flockmtl.py
============================================================================
教学注释 pass (L1 ADD) by Claude.

与 [movie/setup/flockmtl.py] 同结构, 4 个表 (vs movie 2):
  - ben_piazza: 演员事实表
  - ben_piazza_text_data / lizzy_caplan_text_data: 演员相关 wiki 文本表
  - tampa_international_airport: 不相关 (用来测 SQL 多表 JOIN)

默认 model_name 改 "gpt-5-mini" (vs movie 默认 gpt-4o-mini) — 但 gpt-5 系列实际不存在/未上线;
看起来是 dev 时占位 (LOG.md "遗留问题": mmqa 整 scenario 跑不通).

⚠ mmqa scenario 整体 broken:
  - 本 setup: ✓ OK (能创建 conn + load 表)
  - runner [mmqa/runner/flockmtl_runner/flockmtl_runner.py]: ❌ broken
    - import 路径错 `from src.scenario...` (应该 `from scenario...`)
    - 所有 16 个 _execute_q* 方法是 commented-out dead code
  - 缺 SQL 模板: files/mmqa/query/flockmtl/ 整个目录不存在
  → 即使 setup 跑通, runner 没法 dispatch 到任何 query
============================================================================
"""

import os
from pathlib import Path

import duckdb

MMQA_FILES_DIR = os.path.abspath(
    Path(__file__).resolve().parents[4] / "files" / "mmqa" / "data"
)


class FlockMTLMMQASetup:
    # ⚠ 默认 model="gpt-5-mini" — 不存在的模型, 实际跑会 LLM API 报 model_not_found
    def __init__(self, model_name: str = "gpt-5-mini"):
        """
        Initializes the FlockMTL connection using environment variables.
        """

        if os.environ.get("OPENAI_API_KEY") is None:
            raise ValueError("Environment variable OPENAI_API_KEY is not set.")

        self.flockmtl_conn = duckdb.connect(
            os.path.join(MMQA_FILES_DIR, "mmqa.duckdb")
        )

        self.flockmtl_conn.install_extension("flockmtl", repository="community")
        self.flockmtl_conn.load_extension("flockmtl")

        self.flockmtl_conn.execute(
            f"""CREATE SECRET (
                TYPE OPENAI,
                API_KEY '{os.environ.get('OPENAI_API_KEY')}'
            );
            """
        )

        if (
            model_name
            not in self.flockmtl_conn.execute("GET MODELS;")
            .fetchdf()["model"]
            .tolist()
        ):
            self.flockmtl_conn.execute(
                """
                CREATE MODEL(
                    'model_name',
                    'model_name',
                    'openai',
                    {
                        "tuple_format": "json",
                        "batch_size": 32,
                        "model_parameters": {"temperature": 0.7}
                    }
                );
                """.replace(
                    "model_name", model_name
                )
            )

    def _upload_file_to_db(self, csv_path: str, table_name: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found at path: {csv_path}.")

        self.flockmtl_conn.execute(
            f"""
            CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path}');
            """  # noqa: E501
        )

    def setup_data(self, data_dir: str):
        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "ben_piazza.csv"),
            table_name="ben_piazza",
        )

        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "ben_piazza_text_data.csv"),
            table_name="ben_piazza_text_data",
        )

        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "lizzy_caplan_text_data.csv"),
            table_name="lizzy_caplan_text_data",
        )

        self._upload_file_to_db(
            csv_path=os.path.join(data_dir, "tampa_international_airport.csv"),
            table_name="tampa_international_airport",
        )

    def get_connection(self):
        """
        Returns the FlockMTL connection.
        """
        return self.flockmtl_conn
