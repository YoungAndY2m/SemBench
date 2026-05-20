"""
============================================================================
教学注释 (Annotation Pass) — SemBench L1 Code\* cars Q3 (image filter + join)
============================================================================

cars Q3: 找出 "图片显示车未被损坏" 的 VIN list.
Pipeline: scan car_images + car table → join → sem_filter image
          (LLM 看 image_path 判是否未损) → project VIN.
特点: image-modal (depends_on=['image_path']). 与 audio query 不同, image
不在 cars wrapper 的 is_audio 路径 (LOG.md 遗留 #4 magic numbers
[2,5,6,7,9] 不含 3).

注释说明: 本注释 pass 只增加 comment, 不改任何原始代码 (CLAUDE.md §5.5 §D 规则).
"""

import os
import pandas as pd
import palimpzest as pz

from palimpzest.core.lib.schemas import ImageFilepath

# define the schema for the data you read from the file
car_image_data_cols = [
    {"name": "image_path", "type": ImageFilepath, "desc": "The filepath containing the car image"},
    {"name": "car_id", "type": int, "desc": "The integer id for the car"},
    {"name": "image_id", "type": int, "desc": "The integer id for the car image"},
    {"name": "transmission", "type": str, "desc": "The string transmission type"},
    {"name": "vin", "type": str, "desc": "The vehicle identification number"},
]


class MyDataset(pz.IterDataset):
  def __init__(self, id: str, car_df: pd.DataFrame):
    super().__init__(id=id, schema=car_image_data_cols)
    self.car_df = car_df

  def __len__(self):
    return len(self.car_df)

  def __getitem__(self, idx: int):
    # get row from dataframe
    return self.car_df.iloc[idx].to_dict()
  

def run(pz_config, data_dir: str, scale_factor: int = 157376):
    # Load data
    car_images = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/image_car_data_{scale_factor}.csv")) 
    cars = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/car_data_{scale_factor}.csv")) 
    
    # Join before since PZ does not support joins
    tmp_join = cars.join(car_images.set_index('car_id'), on='car_id', how='inner')[['image_path', 'car_id', 'image_id', 'transmission', 'vin']]

    tmp_join = MyDataset(id="my-car-data", car_df=tmp_join)

    # Filter transmission
    tmp_join = tmp_join.filter(lambda row: row['transmission'] == 'Manual')

    # Filter image
    tmp_join = tmp_join.sem_filter('You are given an image of a vehicle or its parts. Return true if car is not damaged.', depends_on=['image_path'])
    tmp_join = tmp_join.project(['vin'])
    candidates = tmp_join.limit(10)

    output = candidates.run(pz_config)
    return output

