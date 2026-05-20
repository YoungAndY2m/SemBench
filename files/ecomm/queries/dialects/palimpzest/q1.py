"""
============================================================================
教学注释 (Annotation Pass) — SemBench L1 Code\* ecomm q1 (text product filter)
============================================================================

ecomm q1: 在 styles_details (e-commerce 产品详情) 上做 text-modal sem_filter
找出 product_id list. 签名: run(pz_config, data_dir) — *二参* 与 cars/medical
*三参* (含 scale_factor) 不一致, 详 [LOG.md 遗留 #7](../../../../../../AllSQPE/Palimpzest/LOG.md).
ecomm 数据集默认 ~44K rows.

数据 load 也不同 cars/medical:
- cars/medical 用 csv (sf_{N} 子目录)
- ecomm 用 parquet (styles_details.parquet / images.parquet)

注释说明: 本注释 pass 只增加 comment, 不改任何原始代码 (CLAUDE.md §5.5 §D 规则).
"""

import os
import pandas as pd
import palimpzest as pz


def run(pz_config, data_dir: str):
    # Load data
    styles_details = pd.read_parquet(
        os.path.join(data_dir, "styles_details.parquet")
    ).rename(
        columns={"id": "product_id"}
    )  # prevent naming conflict with internal Palimpzest 'id' column
    styles_details = pz.MemoryDataset(id="styles_details", vals=styles_details)

    # Filter data
    styles_details = styles_details.sem_filter(
        "The product is a backpack from Reebok",
        depends_on=["productDisplayName", "productDescriptors"],
    )
    styles_details = styles_details.project(["product_id"])

    output = styles_details.run(pz_config)
    return output
