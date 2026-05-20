"""
============================================================================
教学注释 (Annotation Pass) — SemBench L1 Code\* ecomm q2 (image filter)
============================================================================

ecomm q2: image-modal sem_filter (在 images.parquet 上, LLM 看产品图片
判某属性). Like cars Q3 但 ecomm 数据集.

注释说明: 本注释 pass 只增加 comment, 不改任何原始代码 (CLAUDE.md §5.5 §D 规则).
"""

import os
import pandas as pd
import palimpzest as pz


def run(pz_config, data_dir: str):
    # Load data
    images = pz.ImageFileDataset(
        id="images", path=os.path.join(data_dir, "images")
    )

    # Filter data
    images = images.sem_filter(
        "The image shows a (pair of) sports shoe(s) that feature the colors yellow and silver",
        depends_on=["contents"],
    )
    images = images.add_columns(
        udf=lambda row: {"product_id": row["filename"].split(".", 1)[0]},
        cols=[
            {
                "name": "product_id",
                "type": str,
                "description": "Product id generated from image name",
            }
        ],
    )
    images = images.project(["product_id"])

    output = images.run(pz_config)
    return output
