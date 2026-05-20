"""
============================================================================
教学注释 (Annotation Pass) — SemBench L1 Code\* cars Q10 (sem_map classification)
============================================================================

cars Q10: 把每个 complaint 分类到 24 个 problem_category 之一 (LLM 分类任务).
Pipeline: scan + join → sem_map (LLM 看 summary → 输出 problem_category) → project.
特点: 用 sem_map (convert) 而非 sem_filter. cols=[{...}] 指定输出新字段
+ 自然语言 classification 指令. desc 中列出 24 个候选类别让 LLM 闭集选择.

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
    complaints_text = pz.MemoryDataset(id="complaints", vals=tmp_join)

    # Classify
    complaints_text = complaints_text.sem_map(
        cols=[{'name': 'problem_category', 'type': str, 'desc': 'Classify car complaint to one of given problem categories. Answer only one of given problem categories, nothing more. Complaint: Categories: ELECTRICAL SYSTEM, POWER TRAIN, ENGINE, STEERING, SERVICE BRAKES, STRUCTURE, AIR BAGS, ENGINE AND ENGINE COOLING, VEHICLE SPEED CONTROL, VISIBILITY/WIPER, FUEL/PROPULSION SYSTEM, FORWARD COLLISION AVOIDANCE, EXTERIOR LIGHTING, SUSPENSION, FUEL SYSTEM, VISIBILITY, WHEELS, SEAT BELTS, BACK OVER PREVENTION, TIRES, SEATS, LATCHES/LOCKS/LINKAGES, LANE DEPARTURE, EQUIPMENT.'}], 
        depends_on=['summary'])
    complaints_text = complaints_text.project(['car_id', 'problem_category'])
    
    output = complaints_text.run(pz_config)
    return output

