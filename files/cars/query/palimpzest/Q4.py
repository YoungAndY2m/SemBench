"""
============================================================================
教学注释 (Annotation Pass) — SemBench L1 Code\* cars Q4 (text filter on engine problems)
============================================================================

cars Q4: text-modal sem_filter "complaint 提到引擎相关问题" → project year.
Pipeline: scan complaints + car → join → sem_filter (text engine 问题) → project.

注释说明: 本注释 pass 只增加 comment, 不改任何原始代码 (CLAUDE.md §5.5 §D 规则).
"""

import os
import pandas as pd
import palimpzest as pz

def run(pz_config, data_dir: str, scale_factor: int = 157376):
    # Load data
    complaints_text = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/text_complaints_data_{scale_factor}.csv"))
    cars = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/car_data_{scale_factor}.csv"))

    # Join before since PZ does not support joins
    tmp_join = cars.join(complaints_text.set_index('car_id'), on='car_id', how='inner')
    tmp_join = pz.MemoryDataset(id="complaints_cars", vals=tmp_join)

    # Filter and get avg year
    tmp_join = tmp_join.sem_filter('In the complaint, the car has some problems with engine / connected to engine. Complaint:', depends_on=['summary'])
    tmp_join = tmp_join.project(['year'])
    tmp_join = tmp_join.average()

    output = tmp_join.run(pz_config)
    # Compute 2026 - AVG(year)
    avg_year_df = output.to_df()
    avg_year = avg_year_df.iloc[0, 0]
    result_df = pd.DataFrame({'average_age': [2026 - avg_year]})
    # Ensure column name is a string (not Index or other type)
    result_df.columns = [str(col) for col in result_df.columns]
    
    # Return DataFrame directly - runner will handle it
    # Preserve execution stats from the original output
    class ResultWrapper:
        def __init__(self, df, exec_stats):
            self._df = df
            self.execution_stats = exec_stats
        
        def to_df(self):
            return self._df
    
    return ResultWrapper(result_df, output.execution_stats)

