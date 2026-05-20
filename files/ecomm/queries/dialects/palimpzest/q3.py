"""
============================================================================
教学注释 (Annotation Pass) — SemBench L1 Code\* ecomm q3 (sem_add_columns)
============================================================================

ecomm q3: 给每个 product 抽 brand name (text-modal). 用
`sem_add_columns(cols=[{name, type, description}])` API — Palimpzest 的
"add new column" 别名, 等价于 convert + project. 让用户更明确"我要加新字段".

depends_on=['productDisplayName', 'productDescriptors'] = LLM 看这两个字段
推 brand. 输出 (product_id, category=brand_name).

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

    # Perform map/extract
    styles_details = styles_details.sem_add_columns(
        cols=[
            {
                "name": "category",
                "type": str,
                "description": "Extract the brand name from the following product description. Only return the brand name, nothing else.",
            }
        ],
        depends_on=["productDisplayName", "productDescriptors"],
    )
    styles_details = styles_details.project(["product_id", "category"])

    output = styles_details.run(pz_config)
    return output
