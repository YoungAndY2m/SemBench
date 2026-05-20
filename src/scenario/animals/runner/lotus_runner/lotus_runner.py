"""
Created on July 22, 2025

@author: Jiale Lao

LOTUS system runner implementation for animals use case.
Implements Q1, Q3, Q7, Q10 (image-only queries since LOTUS cannot process audio).
"""

# ============================================================
# 教学注释 (L1 wrapper pass):
# ------------------------------------------------------------
# Animals scenario 的 Code mode wrapper (而非 Code*). 与 cars/medical/ecomm
# 不同, 这里 query 都是 *inline* 方法 (_execute_q1 / q3 / q7 / q10), 不动
# 态加载 Q<i>.py 文件.
#
# ⚠ Image-only 限制 (line 7 docstring):
#   原 animals scenario 含 audio (动物叫声) + image 双模态, 但 LOTUS 不
#   能处理音频 (LM provider 大多没 audio modality), 所以 L1 只实现 4 个
#   image-only query (Q1/Q3/Q7/Q10), 缺 Q2/Q4/Q5/Q6/Q8/Q9 (涉及 audio
#   的). 这是 SemBench 团队为了对齐 BigQuery / ThalamusDB (它们能 audio)
#   而强制声明的 LOTUS limitation, 而非 LOTUS 内在不能.
#
# ImageArray 关键用法 (line 56):
#   image_data_df.loc[:, "Image"] = ImageArray(image_data_df["ImagePath"])
#   把字符串路径列包成 LOTUS 自定义 ImageDtype, 让后续 sem_filter 自动
#   走 image multimodal 路径 (df2multimodal_info 检测 ImageDtype 列).
#
# 4 个 query 都是 sem_filter "image contains zebra" + 后续 pandas groupby
# 统计. 不用 cascade (policy=approximate 没 wire RM/VS 启用). 全 oracle LM.
# ============================================================
import pandas as pd
from lotus.dtype_extensions import ImageArray
from pathlib import Path

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_lotus_runner.generic_lotus_runner import GenericLotusRunner


class LotusRunner(GenericLotusRunner):
    """Runner for LOTUS system for animals use case."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        """
        Initialize LOTUS runner for animals.

        Args:
            use_case: The use case to run (should be 'animals')
            model_name: LLM model to use
        """
        super().__init__(
            use_case, scale_factor, model_name, concurrent_llm_worker
        )

    # ========================================================
    # _execute_q1: 最简单 query — 全表 sem_filter + count
    # --------------------------------------------------------
    # 流程:
    #   1) load_data 拿 image_data.csv (ImagePath / City / StationID 等列)
    #   2) ImageArray 包 ImagePath 列 → df["Image"] 成 ImageDtype
    #   3) sem_filter "image contains zebra" → bool, 留 True 行
    #   4) len(filtered) 给 count
    #
    # SemBench 输出要求每个 query 返 DataFrame, 这里 wrap 单 row 含
    # count(*) 列. 与 SQL "SELECT COUNT(*)" 输出对齐.
    # ========================================================
    def _execute_q1(self) -> pd.DataFrame:
        """
        Execute Q1: Count the number of pictures of zebras.

        This uses semantic filtering to identify zebra images and counts them.

        Returns:
            DataFrame with columns: count(*)
        """
        # 1. Load the image data
        image_data_df = self.load_data("image_data.csv")

        # 2. Wrap the image path column in ImageArray
        image_data_df.loc[:, "Image"] = ImageArray(image_data_df["ImagePath"])

        # 3. Perform semantic filtering to identify zebra images
        filter_instruction = "The image {Image} contains a zebra."
        zebra_images = image_data_df.sem_filter(filter_instruction)

        # 4. Count the number of zebra images
        count = len(zebra_images)

        # 5. Return result in standardized format
        result = pd.DataFrame([{"count(*)": count}])

        return result

    def _execute_q3(self) -> pd.DataFrame:
        """
        Execute Q3: Find the city where we captured most pictures of zebras.

        Returns:
            DataFrame with columns: city
        """
        # 1. Load the image data
        image_data_df = self.load_data("image_data.csv")

        # 2. Wrap the image path column in ImageArray
        image_data_df.loc[:, "Image"] = ImageArray(image_data_df["ImagePath"])

        # 3. Perform semantic filtering to identify zebra images
        filter_instruction = "The image {Image} contains a zebra."
        zebra_images = image_data_df.sem_filter(filter_instruction)

        # 4. Group by city and count, then get the city with most zebras
        if len(zebra_images) == 0:
            # If no zebras found, return empty result
            result = pd.DataFrame(columns=["city"])
        else:
            city_counts = (
                zebra_images.groupby("City").size().reset_index(name="count")
            )
            city_counts = city_counts.sort_values("count", ascending=False)

            # Get the city with most zebra pictures (ties broken arbitrarily by taking first)
            top_city = city_counts.iloc[0]["City"]
            result = pd.DataFrame([{"city": top_city}])

        return result

    # ========================================================
    # _execute_q7: 双 sem_filter + set intersection
    # --------------------------------------------------------
    # ⚠ 关键 quirk (line 122): 第二次 sem_filter 前重新 wrap ImageArray.
    # 因为第一次 sem_filter 内部可能 invalidate cache 或修改 df, 重 wrap
    # 是防御性编程. 实际 ImageArray.copy() 应该足够, 这里多一次保险.
    #
    # 算法: 两次独立 sem_filter (zebra + impala), 各自拿 City set, 求交集.
    # 用 pandas set 而非 sem_filter 嵌套 → 更直观, 但 cost 2× LLM 调用.
    # ========================================================
    def _execute_q7(self) -> pd.DataFrame:
        """
        Execute Q7: Find cities where zebras and impalas co-occur in images.

        Returns:
            DataFrame with columns: city
        """
        # 1. Load the image data
        image_data_df = self.load_data("image_data.csv")

        # 2. Wrap the image path column in ImageArray
        image_data_df.loc[:, "Image"] = ImageArray(image_data_df["ImagePath"])

        # 3. Perform semantic filtering to identify zebra images
        zebra_filter = "The image {Image} contains a zebra."
        zebra_images = image_data_df.sem_filter(zebra_filter)

        # 4. Perform semantic filtering to identify impala images
        # Reset image data for second filter
        image_data_df.loc[:, "Image"] = ImageArray(image_data_df["ImagePath"])

        impala_filter = "The image {Image} contains an impala."
        impala_images = image_data_df.sem_filter(impala_filter)

        # 5. Find intersection of cities with zebras and cities with impalas
        zebra_cities = (
            set(zebra_images["City"]) if len(zebra_images) > 0 else set()
        )
        impala_cities = (
            set(impala_images["City"]) if len(impala_images) > 0 else set()
        )

        cooccur_cities = zebra_cities.intersection(impala_cities)

        # 6. Return result
        if cooccur_cities:
            result = pd.DataFrame(
                [{"city": city} for city in sorted(cooccur_cities)]
            )
        else:
            result = pd.DataFrame(columns=["city"])

        return result

    def _execute_q10(self) -> pd.DataFrame:
        """
        Execute Q10: Find the city and station with most associated pictures showing zebras.

        Returns:
            DataFrame with columns: city, stationID
        """
        # 1. Load the image data
        image_data_df = self.load_data("image_data.csv")

        # 2. Wrap the image path column in ImageArray
        image_data_df.loc[:, "Image"] = ImageArray(image_data_df["ImagePath"])

        # 3. Perform semantic filtering to identify zebra images
        filter_instruction = "The image {Image} contains a zebra."
        zebra_images = image_data_df.sem_filter(filter_instruction)

        # 4. Group by city and station, count, then get the pair with most zebras
        if len(zebra_images) == 0:
            # If no zebras found, return empty result
            result = pd.DataFrame(columns=["city", "stationID"])
        else:
            station_counts = (
                zebra_images.groupby(["City", "StationID"])
                .size()
                .reset_index(name="count")
            )
            station_counts = station_counts.sort_values(
                "count", ascending=False
            )

            # Get the city/station pair with most zebra pictures (ties broken arbitrarily)
            top_row = station_counts.iloc[0]
            result = pd.DataFrame(
                [{"city": top_row["City"], "stationID": top_row["StationID"]}]
            )

        return result

    def _get_empty_results_dataframe(self, query_id: int) -> pd.DataFrame:
        """
        Get empty DataFrame with correct columns for a query.

        Args:
            query_id: ID of the query

        Returns:
            Empty DataFrame with correct columns
        """
        if query_id == 1:
            return pd.DataFrame(columns=["count(*)"])
        elif query_id == 3:
            return pd.DataFrame(columns=["city"])
        elif query_id == 7:
            return pd.DataFrame(columns=["city"])
        elif query_id == 10:
            return pd.DataFrame(columns=["city", "stationID"])
        else:
            return pd.DataFrame()
