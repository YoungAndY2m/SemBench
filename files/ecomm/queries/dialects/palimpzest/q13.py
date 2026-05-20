"""
============================================================================
教学注释 (Annotation Pass) — SemBench L1 Code\* ecomm q13 (text sem_filter)
============================================================================

ecomm q13: text-modal sem_filter on styles_details — ecomm 系列最后一个
query. 不同谓词 / 不同 retrieve 字段.

注释说明: 本注释 pass 只增加 comment, 不改任何原始代码 (CLAUDE.md §5.5 §D 规则).
"""

import os
import pandas as pd
import palimpzest as pz


def add_image_data(
    styles_details: pz.MemoryDataset, data_dir: str
) -> pz.MemoryDataset:
    styles_details = styles_details.add_columns(
        udf=lambda row: {
            "image_file_path": os.path.join(
                data_dir, "images", str(row["product_id"]) + ".jpg"
            )
        },
        cols=[
            {
                "name": "image_file_path",
                "type": pz.ImageFilepath,
                "description": "",  # leave empty because this influences the LLMs decision
            }
        ],
    )
    return styles_details


def run(pz_config, data_dir: str):
    # Load data
    styles_details = pd.read_parquet(
        os.path.join(data_dir, "styles_details.parquet")
    ).rename(
        columns={"id": "product_id"}
    )  # prevent naming conflict with internal Palimpzest 'id' column
    
    styles_details = pz.MemoryDataset(id="styles_details", vals=styles_details)
    styles_details = add_image_data(styles_details, data_dir)

    # Filter data
    styles_details = styles_details.sem_filter(
        """
        You will receive a description of what a customer is looking for together with an image and a textual description of the product.
        Determine if they both match.
    
        I am looking for a running shirt for men with a round neck and short sleeves,
        preferably in blue or black, but not bright colors like white.
        Also definitely not green.
        It should be suitable for outdoor running in warm weather.
        If the t-shirt is not green, it should at least feature a striped design.

        The product has the following image and textual description:
        """,
        depends_on=[
            "productDisplayName",
            "productDescriptors",
            "image_file_path",
        ],
    )
    
    styles_details = styles_details.project(["product_id"])

    output = styles_details.run(pz_config)
    return output
