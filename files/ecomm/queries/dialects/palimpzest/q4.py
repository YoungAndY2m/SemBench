"""
============================================================================
教学注释 (Annotation Pass) — SemBench L1 Code\* ecomm q4 (image sem_add_columns)
============================================================================

ecomm q4: 同 q3 模式但走 image — `sem_add_columns` 给每个 product
based on image 抽 category. Image-modal classification.

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
    styles_details = pd.read_parquet(
        os.path.join(data_dir, "styles_details.parquet")
    )

    # Pre-filter for simple colors
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
    styles_details = styles_details[
        styles_details["baseColour"].isin(
            ["Black", "Blue", "Red", "White", "Orange", "Green"]
        )
    ]
    images = images.filter(
        lambda row: int(row["product_id"]) in styles_details["id"].values
    )

    # Process data
    images = images.sem_add_columns(
        cols=[
            {
                "name": "category",
                "type": str,
                "description": "Extract the primary color of the product in the image. Only return the base color, nothing else.",
            }
        ],
        depends_on=["contents"],
    )
    images = images.project(["product_id", "category"])

    output = images.run(pz_config)
    return output
