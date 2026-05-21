"""
============================================================================
教学注释 (Annotation Pass) — Medical scenario gold-SQL 验证脚本
============================================================================
本文件不是 module, 是 *独立验证脚本* (跑一遍打印 Q1-Q7 是否通过). 作用:
- 把 4 个 medical CSV (patient / audio / image / text) 加载到 in-memory DuckDB
- 对 Q1-Q7 各自跑 *两条* 等价计算:
  1. SQL 路径: `con.execute(<SQL>).fetchdf()` 走 DuckDB
  2. pandas 路径: 直接对 `patient_df` 做 boolean indexing
- 比对两条结果是否一致 (.equals / .isin / .values 对比), print 结果 (True/False)

这是 paper 作者写 gold SQL 时的自检脚本: 确保 SQL 语义跟 pandas 等价
(避免 gold SQL 错了, 害下游 evaluator 比错). 跑一次手动看 print 输出.

★ 注意几处 SQL bug / 不一致:
- Q3: `did_family_have_cancer == 1` 用了 Python `==` 不是 SQL `=` (DuckDB 容忍
  Python 风格); LIMIT 5 但 pandas 路径没 limit
- Q7: SQL `FROM patient_df` (用 pandas 变量名当表名) — 走 DuckDB 的"Python
  fragment integration" (自动识别外部 df 变量); 跟其它 Q 的 'patients' 不一致
- Q3 比较用 `.isin(gt).all().all()` 不是 `.equals(gt)`, 因为 LIMIT 让两边不
  完全相同, 只要 SQL 结果是 gt 的子集就 OK

注释说明: 本注释 pass 只增加 comment, 不改任何原始代码.
============================================================================
"""

from pathlib import Path

import numpy as np
import duckdb
import pandas as pd

# In-memory DB (no file)
con = duckdb.connect()

data_path = Path("files/medical/data")

patient_df = pd.read_csv(data_path / "patient_data_with_labels.csv")
audio_df = pd.read_csv(data_path / "audio_lung_data.csv")
image_x_ray_df = pd.read_csv(data_path / "image_x_ray_data.csv")
symptoms_text_df = pd.read_csv(data_path / "text_symptoms_data.csv")

con.register("patients", patient_df)
con.register("audio", audio_df)
con.register("image", image_x_ray_df)
con.register("text", symptoms_text_df)

# Q1
# Which patients have symptoms of allergy in our database?
q1 = con.execute(
    "SELECT age, gender, smoking_history, did_family_have_cancer, patient_id FROM patients WHERE text_diagnosis = 'allergy'"
).fetchdf()

sick = patient_df[patient_df["text_diagnosis"] == "allergy"]
gt = sick[
    ["age", "gender", "smoking_history", "did_family_have_cancer", "patient_id"]
].copy()

q1 = q1.sort_values(by="patient_id").reset_index(drop=True)
gt = gt.sort_values(by="patient_id").reset_index(drop=True)

print("Q1", q1.equals(gt), (set(sick["patient_id"]) - set(gt["patient_id"])) == set())

# Q2
# Find patients with available audio sound who are non-smokers (never smoked or stopped smoking) and healthy considering the sound recordings in our database.
q2 = con.execute(
    "SELECT age, gender, smoking_history, did_family_have_cancer, patient_id "
    "FROM patients "
    "WHERE smoking_history != 'Current' AND audio_diagnosis = 'normal'"
).fetchdf()

healthy = patient_df[
    (patient_df["smoking_history"] != "Current")
    & (patient_df["audio_diagnosis"] == "normal")
]
gt = healthy[
    ["age", "gender", "smoking_history", "did_family_have_cancer", "patient_id"]
].copy()

q2 = q2.sort_values(by="patient_id").reset_index(drop=True)
gt = gt.sort_values(by="patient_id").reset_index(drop=True)

print("Q2", q2.equals(gt))

# Q3
# Find five patients who have cancer in their family history and have lung problems according to their X-ray images (considered sick).
q3 = con.execute(
    "SELECT age, gender, smoking_history, did_family_have_cancer, patient_id "
    "FROM patients "
    "WHERE did_family_have_cancer == 1 AND x_ray_diagnosis NOT IN ('00_normal', 'none')"
    "LIMIT 5"
).fetchdf()


had_cancer = (patient_df["did_family_have_cancer"] == 1) & (
    patient_df["x_ray_diagnosis"].isin(["00_normal", "none"]) == False
)
gt = patient_df.loc[
    had_cancer,
    [
        "age",
        "gender",
        "smoking_history",
        "did_family_have_cancer",
        "patient_id",
    ],
]

q3 = q3.sort_values(by="patient_id").reset_index(drop=True)
gt = gt.sort_values(by="patient_id").reset_index(drop=True)

print("Q3", q3.isin(gt).all().all())

# Q4
# What is an average age of patients with acne?
q4 = con.execute(
    "SELECT AVG(age) AS average_acne_age "
    "FROM patients "
    "WHERE text_diagnosis = 'acne'"
).fetchdf()

avg_age = patient_df.loc[patient_df["text_diagnosis"] == "acne", "age"].mean()
gt = pd.DataFrame({"average_acne_age": [avg_age]})

print("Q4", q4.equals(gt))

# Q5
# Count smoking history counts of patients whose audio and X-Ray image suggest lung diseases.
q5 = con.execute(
    """
    SELECT smoking_history, COUNT(*) AS count 
    FROM patients 
    WHERE smoking_history = "Never" AND audio_diagnosis NOT IN ('none', 'normal') AND 
    x_ray_diagnosis NOT IN ('none', '00_normal') 
    GROUP BY smoking_history
    """
).fetchdf()

gt = (
    patient_df.loc[
        (patient_df["audio_diagnosis"].isin(["none", "normal"]) == False)
        & (patient_df["x_ray_diagnosis"].isin(["none", "00_normal"]) == False),
        "smoking_history",
    ]
    .value_counts()
    .to_frame()
)
gt = gt.reset_index()

q5 = q5.sort_values(by="smoking_history").reset_index(drop=True)
gt = gt.sort_values(by="smoking_history").reset_index(drop=True)

print("Q5", q5.equals(gt))

# Q6
# Find the youngest patient who is sick according to at least one modality but healthy according to another modality.
q6 = con.execute(
    "SELECT age, gender, smoking_history, did_family_have_cancer, patient_id "
    "FROM patients WHERE is_sick = TRUE "
    "AND (text_diagnosis IN ('normal', '00_normal') "
    "OR audio_diagnosis IN ('normal', '00_normal') "
    "OR x_ray_diagnosis IN ('normal', '00_normal')) "
    "ORDER BY age ASC "
    "LIMIT 1"
).fetchdf()

sick_patients = patient_df[patient_df["is_sick"]]

gt = sick_patients[
    (
        patient_df["text_diagnosis"]
        + ", "
        + patient_df["audio_diagnosis"]
        + ", "
        + patient_df["x_ray_diagnosis"]
    ).apply(
        lambda x: (x.split(", ").count("normal") > 0)
        | (x.split(", ").count("00_normal") > 0)
    )
]

gt = gt.sort_values(by="age", ascending=True).head(1)[
    [
        "age",
        "gender",
        "smoking_history",
        "did_family_have_cancer",
        "patient_id",
    ]
]

print("Q6", (q6.values == gt.values).all())

# Q7
# What is the average age of a patient with any sickness in out database (consider all symptoms, x-ray images, and lung audio recordings)?
q7 = con.execute(
    "SELECT age, gender, smoking_history, did_family_have_cancer, patient_id FROM patient_df WHERE is_sick = 1"
).fetchdf()

gt = patient_df.loc[
    patient_df["is_sick"] == 1,
    [
        "age",
        "gender",
        "smoking_history",
        "did_family_have_cancer",
        "patient_id",
    ],
]

q7 = q7.sort_values(by="patient_id").reset_index(drop=True)
gt = gt.sort_values(by="patient_id").reset_index(drop=True)

print("Q7", q7.isin(gt).all().all())

print(con.execute(
    "SELECT AVG(age) FROM patient_df WHERE is_sick = 1"
).fetchdf())
