"""
============================================================================
教学注释 (Annotation Pass) — Ecomm scenario 的 ThalamusDB 数据初始化
============================================================================
跟 cars/setup/thalamusdb.py 不同, 本文件 *不装 FlockMTL extension*, 只做最简
data load (DuckDB 原生 read parquet); 因为 ecomm scenario 走 ThalamusDB 时,
extension 由别处管 (可能由 scenario_handler 或 runtime config 处理).

做 2 件事:
1. 删旧 db 文件 + 新建 DuckDB connection
2. 从 parquet 加载 styles_details + image_mapping 两个表,
   styles_details 多算一列 full_product_description (productDisplayName 拼
   description.value), 用来给 LLM semantic filter 做单字符串 input
3. image_mapping 多算一列 local_image_path (绝对路径), 给视觉模型加载图片用

★ 注释里有 2 条 surface 的 ThalamusDB 限制 (paper-level):
   - "cannot execute semantic filters on expressions or multiple columns"
     → 必须先把多列拼接 materialize 成单列
   - "cannot deal with columns containing strings with single quotes"
     → 必须 replace('\\'', '') 去掉单引号
============================================================================
"""

import os
import duckdb
import pandas as pd


class ThalamusDBEcommSetup:
    def setup_data(self, data_dir: str):
        # 幂等: 每次 setup 删旧 db 重建, 避免 schema 漂移.
        db_path = os.path.join(data_dir, "thalamusdb.duckdb")
        if os.path.exists(db_path):
            os.remove(db_path)

        con = duckdb.connect(db_path)
        # file_search_path 让后续 SQL 里写 'styles_details.parquet' (相对路径)
        # 能找到 data_dir 下的文件; DuckDB 自动 join base path.
        con.execute(f"set file_search_path = '{data_dir}'")
        con.execute(
            f"""CREATE OR REPLACE TABLE styles_details AS
            SELECT 
              *,
              -- ThalamusDB cannot execute semantic filters on expressions or multiple columns, so we have to manually concatenate and materialize them.
              -- Further, ThalamusDB cannot deal with columns containing strings with single quotes, so we remove them.
              replace(productDisplayName || ' ' || productDescriptors.description.value, '''', '') AS full_product_description
            FROM 'styles_details.parquet'
            """
        )
        con.execute(
            f"""CREATE OR REPLACE TABLE image_mapping AS
            SELECT
              *,
              '{os.path.join(data_dir, 'images', '')}' || filename AS local_image_path
            FROM 'image_mapping.parquet'
            """
        )
        con.close()
