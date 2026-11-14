# Text-to-SQL Model Fine-Tuning: Results & Benchmark Report

**Model**: `Qwen/Qwen2.5-Coder-3B-Instruct`  
**Fine-Tuning Method**: QLoRA (4-bit NF4, double quant, bfloat16, LoRA $r=16, \alpha=32$)  
**Hardware**: NVIDIA GeForce RTX 3060 12GB, 32GB RAM, Intel i5, Windows 11  
**Benchmark**: 100 Fixed Spider Dev Questions across 18 SQLite Databases (Seed: 42)

---

## 1. Executive Metrics Comparison

| Metric                      | Base Model (Untuned 3B) | Fine-Tuned Model (1-Epoch QLoRA) |             Delta / Impact              |
| :-------------------------- | :---------------------: | :------------------------------: | :-------------------------------------: |
| **Execution Accuracy (EX)** |   **61.0%** (61/100)    |        **59.0%** (59/100)        |  -2.0% (within sampling noise margin)   |
| **Valid-SQL Rate**          |   **86.0%** (86/100)    |        **95.0%** (95/100)        | **+9.0%** (64% reduction in broken SQL) |
| **Avg Latency per Query**   |       **1.911 s**       |           **5.510 s**            |                +3.599 s                 |
| **Training Samples**        |            -            |        5,000 ChatML pairs        | Filtered by in-memory SQLite execution  |
| **Peak VRAM Detected**      |            -            |           **4.00 GB**            |    Well within 12 GB hardware budget    |
| **Training Time**           |            -            |        **30.97 minutes**         |      312 steps, final loss: 0.5100      |

---

## 2. Failure Analysis (Patterns Across Unseen Spider Dev)

An analysis of 15 incorrect query predictions categorized in `results/failure_analysis.json` revealed 4 distinct error categories:

1. **Schema Linkage / Missing JOINs (6 cases / 40.0%)**:
   - The model referenced columns from related tables without performing the requisite JOIN (e.g., querying `Template_Type_Code` directly on `Documents` rather than joining with `Templates`).
2. **Complex Set & Aggregation Logic (5 cases / 33.3%)**:
   - Substituting `OR` or `IN` where set intersection (`INTERSECT`) was required for simultaneous constraints across multiple rows, or substituting `SUM` on percentages for `COUNT(*)`.
3. **Column Projection Order (2 cases / 13.3%)**:
   - The model selected the correct columns and values, but in a different order than the gold reference query (e.g., `SELECT Population, Name...` vs gold `SELECT Name, Population...`), resulting in a positional tuple mismatch.
4. **Condition Literal Mismatch (2 cases / 13.3%)**:
   - String literal formatting discrepancy (e.g., filtering `Citizenship != 'French'` vs gold `Citizenship != 'France'`, or case-sensitive string matching).

---
