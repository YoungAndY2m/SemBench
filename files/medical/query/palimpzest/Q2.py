"""
============================================================================
教学注释 (Annotation Pass) — SemBench L1 Code\* medical Q2 (audio lungs healthy filter)
============================================================================

medical Q2: scan audio lungs + patient → join → sem_filter audio
(LLM 听判 healthy lungs without disease).
注释中带 FIXME: ".distinct()" 因 PZ 的 cost 字段 issue 暂未用; 后续输出
DataFrame 上手动 drop_duplicates 但会丢 cost 数据.

注释说明: 本注释 pass 只增加 comment, 不改任何原始代码 (CLAUDE.md §5.5 §D 规则).
"""

import os
import pandas as pd
import palimpzest as pz
from palimpzest.core.lib.schemas import AudioFilepath

# define the schema for the data you read from the file
data_cols = [
    {"name": "path", "type": AudioFilepath, "desc": "The filepath containing audios."},
    {"name": "patient_id", "type": int, "desc": "The integer id for the patient."},
    {"name": "audio_id", "type": int, "desc": "The integer id for the audio."},
    {"name": "smoking_history", "type": str, "desc": "The string smoking_history."},
]

class MyDataset(pz.IterDataset):
  def __init__(self, id: str, x_ray_df: pd.DataFrame):
    super().__init__(id=id, schema=data_cols)
    self.x_ray_df = x_ray_df

  def __len__(self):
    return len(self.x_ray_df)

  def __getitem__(self, idx: int):
    # get row from dataframe
    return self.x_ray_df.iloc[idx].to_dict()


def run(pz_config, data_dir: str, scale_factor: int = 11112):
    # Load data
    audio = pd.read_csv(os.path.join(data_dir, "data/audio_lung_data.csv" if scale_factor == 11112 else f"data/audio_lung_data_{scale_factor}.csv"))
    patients = pd.read_csv(os.path.join(data_dir, "data/patient_data.csv" if scale_factor == 11112 else f"data/patient_data_{scale_factor}.csv")) 
    
    # Join
    tmp_join = patients.join(audio.set_index('patient_id'), on='patient_id', how='inner')[['path', 'patient_id', 'audio_id', 'smoking_history']]

    tmp_join = MyDataset(id="my-data", x_ray_df=tmp_join)

    # Filter 
    tmp_join = tmp_join.filter(lambda row: row['smoking_history'] != 'Current')

    # Filter sickness
    tmp_join = tmp_join.sem_filter('You are given an audio recording of human lungs from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Return true if the recording is healthy lungs, without diseases.', depends_on=['path'])
    tmp_join = tmp_join.project(['patient_id'])

    # FIXME get distinct values  but output.to_df().drop_duplicates() returns only df w/o cost
    seen = set()
    output = tmp_join.filter(lambda row: row not in seen and not seen.add(tuple(row)))
    
    output = tmp_join.run(pz_config)
    return output
