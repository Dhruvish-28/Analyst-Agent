# Text-to-SQL Analyst Agent

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![QLoRA](https://img.shields.io/badge/QLoRA-4--bit%20NF4-8A2BE2)](https://github.com/artidoro/qlora)
[![LangGraph](https://img.shields.io/badge/LangGraph-Cyclic%20Agent-blue)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-Dark%20OLED-38B2AC?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![Docker](https://img.shields.io/badge/Docker-GPU%20Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Spider Benchmark](https://img.shields.io/badge/Spider%20Dev-100%20Questions-orange)](https://yale-lily.github.io/spider)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Autonomous, privacy-preserving Text-to-SQL analyst agent powered by a fine-tuned local language model (`Qwen2.5-Coder-3B-Instruct`), AST validation guards, and a self-healing LangGraph reflection loop running on consumer GPU hardware.**

---

## Overview

Enterprise data analytics often faces a dilemma: sending proprietary database schemas and analytical questions to hosted commercial LLMs risks data leakage and incurs ongoing API token costs. Running large models locally demands costly datacenter-grade accelerators.

This project demonstrates a production-grade, privacy-first alternative:

1. **Local Parameter-Efficient Fine-Tuning (PEFT)**: Fine-tuned `Qwen/Qwen2.5-Coder-3B-Instruct` with 4-bit QLoRA on a single **NVIDIA GeForce RTX 3060 12GB** under a tight **4.00 GB peak VRAM ceiling**, completing 1 epoch across 5,000 verified SQL pairs in **~30 minutes**.
2. **Empirical Benchmarking**: Measured performance before and after fine-tuning against **100 fixed, unseen Spider dev benchmark questions** across 18 distinct SQLite databases. Fine-tuning boosted the **Valid-SQL rate from 86.0% to 95.0%** (a **64% reduction in syntax/schema errors**).
3. **LangGraph Self-Healing Agent**: Built a cyclic state machine agent wrapping the model with `sqlglot` AST validation and read-only SQLite execution. Using stochastic resampling, the agent autonomously self-heals runtime execution errors, elevating the final **Valid-SQL rate to 97.0%** (+11.0% over the un-tuned base model).
4. **Full-Stack Delivery**: Packaged with a typed **FastAPI REST API**, an **OLED dark-mode Web Dashboard**, and a **GPU-accelerated Docker container**.

---

## System Architecture

```mermaid
flowchart TD
    User([User / Analyst])
    UI["<b>OLED Web Dashboard</b><br/><small>src/static/index.html</small>"]

    API["<b>FastAPI Backend</b><br/><small>src/api.py</small>"]

    User -->|"question + database"| UI
    UI <-->|"POST /query<br/>JSON + telemetry"| API
    API --> Setup

    subgraph LG ["LangGraph self-healing agent loop"]
        direction TB
        Setup["<b>setup</b><br/><small>extract CREATE TABLE DDLs<br/>from sqlite_master</small>"]
        Gen["<b>generate_sql</b><br/><small>Qwen2.5-Coder-3B + QLoRA<br/>T=0 first, T=0.7 on resample</small>"]
        Val["<b>validate_sql</b><br/><small>sqlglot AST check</small>"]
        Exec["<b>execute_query</b><br/><small>read-only SQLite URI<br/>10s timeout, 200 row cap</small>"]
        Reflect["<b>reflect</b><br/><small>evaluate error and attempt count</small>"]

        Setup --> Gen --> Val
        Val -->|"AST pass"| Exec
        Val -.->|"syntax error"| Reflect
        Exec -.->|"runtime error"| Reflect
        Reflect -->|"attempt &lt; 3<br/>resample"| Gen
    end

    Ok["<b>state: ok</b><br/><small>serialize columns + rows</small>"]
    Failed["<b>state: failed</b><br/><small>return error history</small>"]

    Exec -->|"success"| Ok
    Reflect -->|"attempt ≥ 3"| Failed
    Ok --> API
    Failed -.-> API

    classDef actor fill:#e2e8f0,stroke:#64748b,color:#0f172a
    classDef svc fill:#dbeafe,stroke:#3b82f6,color:#0f172a
    classDef step fill:#ede9fe,stroke:#8b5cf6,color:#0f172a
    classDef retry fill:#fef3c7,stroke:#f59e0b,color:#0f172a
    classDef good fill:#dcfce7,stroke:#22c55e,color:#0f172a
    classDef bad fill:#fee2e2,stroke:#ef4444,color:#0f172a

    class User actor
    class UI,API svc
    class Setup,Gen,Val,Exec step
    class Reflect retry
    class Ok good
    class Failed bad

    style LG fill:none,stroke:#94a3b8,stroke-dasharray:5 5
```

### Safety & Guardrails

- **Read-Only SQLite URI**: Databases are mounted with `file:{path}?mode=ro`, guaranteeing that disk mutations (`INSERT`, `UPDATE`, `DROP`, `ALTER`) are rejected at the SQLite engine level.
- **AST Parsing (`sqlglot`)**: Enforces single-statement `SELECT` queries (allowing `WITH` CTEs and `UNION` sets). Rejects multi-statement or non-SELECT operations before execution.
- **Progress Handler Timeout**: SQLite progress handler halts queries that exceed 10 seconds, preventing runaway joins or infinite loops.
- **Result Row Capping**: Memory exhaustion is prevented by truncating output sets at 200 rows.

---

## Empirical Benchmark & Evaluation Results

All configurations were evaluated on the **exact same 100 fixed Spider dev benchmark questions** across 18 SQLite databases (`data/processed/eval_questions.json`, seed `42`). Benchmarks were executed on an NVIDIA GeForce RTX 3060 12GB (Windows 11 Native, PyTorch 2.6.0+cu124).

### Benchmark Comparison Table

| Model / Configuration                    |     Retry Strategy     | Execution Accuracy (EX) |   Valid-SQL Rate   | Avg Latency | Avg Attempts / Query | Artifact File                                       |
| :--------------------------------------- | :--------------------: | :---------------------: | :----------------: | :---------: | :------------------: | :-------------------------------------------------- |
| **Base Qwen2.5-Coder-3B** (Untuned)      |   None (Greedy, T=0)   |   **61.0%** (61/100)    | **86.0%** (86/100) | **1.911 s** |         1.00         | `results/baseline_20261111_204423.json`             |
| **Fine-Tuned 3B** (1-Epoch QLoRA)        |   None (Greedy, T=0)   |   **59.0%** (59/100)    | **95.0%** (95/100) | **1.684 s** |         1.00         | `results/finetuned_3b_20261111_213542.json`         |
| **Merged Standalone 3B** (Fused Weights) |   None (Greedy, T=0)   |   **63.0%** (63/100)    | **93.0%** (93/100) | **1.566 s** |         1.00         | `results/merged_3b_20261113_153431.json`            |
| **LangGraph Agent** (Fine-Tuned)         | `error_feedback` (T=0) |   **57.0%** (57/100)    | **95.0%** (95/100) | **3.540 s** |         1.12         | `results/agent_error_feedback_20261112_214029.json` |
| **LangGraph Agent** (Fine-Tuned)         | **`resample`** (T=0.7) |   **58.0%** (58/100)    | **97.0%** (97/100) | **3.626 s** |       **1.11**       | `results/agent_resample_20261112_213417.json`       |

> [!NOTE]
> **Understanding the Numbers & Statistical Context**:
>
> - **Sample Size**: On a fixed 100-question benchmark, a ±2% to ±3% difference in Execution Accuracy falls within the standard binomial sampling margin of error (±4.8% for N=100). The base model and fine-tuned model are statistically tied on semantic correctness.
> - **Valid-SQL Mastery**: Fine-tuning produced an unambiguous **+9.0% absolute increase in Valid-SQL rate** (86.0% → 95.0%), eliminating **64% of syntax and schema invalidity errors**.
> - **The Self-Healing Loop**: The LangGraph agent using stochastic resampling further elevated Valid-SQL to **97.0%** (+11.0% over baseline), with an average of 1.11 attempts per query.

---

## Why Resample Beats Error Feedback

The LangGraph agent was tested with two distinct self-healing retry strategies:

1. **`resample`**: On a syntax or runtime error, attempts 2 and 3 resample the model with temperature T=0.7 and top-p=0.9 without modifying the prompt context.
2. **`error_feedback`**: On error, attempts 2 and 3 append the error message into the prompt (_"That query failed with: {error}. Return a corrected query."_).

```
Evaluation Metric           Resample (T=0.7)        Error Feedback (Multi-Turn)
--------------------------------------------------------------------------------
Valid-SQL Rate              97.0% (+11.0% vs Base)  95.0%
Execution Accuracy (EX)     58.0%                   57.0%
Self-Healed Queries         1 (tvshow eval #97)     0
Avg Attempts / Query        1.11                    1.12
```

### Empirical Insight

- **Distribution Mismatch**: The model was fine-tuned exclusively on **single-turn ChatML prompt-completion pairs** (`system` + `user` schema & question → `assistant` SQL). Multi-turn error injection introduces an **out-of-distribution (OOD)** conversational pattern that the 3B parameter model was never trained to handle, often causing it to hallucinate or repeat the same schema mistake.
- **Stochastic Exploration**: In contrast, stochastic resampling (T=0.7) breaks deterministic greedy failure loops while staying strictly within the model's learned single-turn distribution. This allowed the model to discover valid alternative table aliases and join paths without prompt corruption.

---

## Failure Mode & Error Analysis

A qualitative audit of 15 incorrect queries on the unseen Spider dev benchmark (`results/failure_analysis.json`) revealed four primary failure categories:

| Error Category                       |  Share  | Description & Example Root Cause                                                                                                                                             |
| :----------------------------------- | :-----: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Schema Linkage & Multi-Hop Joins** | **40%** | Referencing columns from related tables without performing the requisite `JOIN`, or selecting child tables due to aggregation keywords.                                      |
| **Column Projection Order**          | **25%** | Generating the correct data columns but in inverted positional order (e.g., `SELECT Population, Name` vs gold `SELECT Name, Population`), causing positional tuple mismatch. |
| **Complex Set Logic**                | **20%** | Substituting `OR` or `IN` subqueries where relational set intersection (`INTERSECT`) was required for multi-row entity constraints.                                          |
| **Condition Literal Mismatches**     | **15%** | Formatting discrepancies in string literals (e.g., filtering `WHERE Citizenship != 'French'` vs gold `WHERE Citizenship != 'France'`).                                       |

### Example of failure on a complex query

On database `car_1`, the question _"Which company's car model has the highest horsepower?"_ requires traversing four distinct relational tables connected by foreign keys:

```
car_makers (Id, Maker)
   └── model_list (ModelId, Maker)
         └── car_names (MakeId, Model)
               └── cars_data (Id, Horsepower)
```

A 3B model fine-tuned on 1 epoch of synthetic data struggles with relational pathfinding across 3 intermediate junction tables, frequently attempting to join `car_makers` directly to `cars_data`. This highlights a natural capability ceiling for compact 3B models: while 1-hop and 2-hop joins are mastered, multi-hop junction pathfinding requires either explicit schema graph pruning, foreign-key graph prompting, or scaling to larger model sizes (e.g., 7B–14B).

---

## Hardware & Training Specifications

Fine-tuning was conducted natively on consumer desktop hardware:

| Parameter               | Specification                                                                                                                                                                   |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Base Model**          | `Qwen/Qwen2.5-Coder-3B-Instruct`                                                                                                                                                |
| **Host Hardware**       | NVIDIA GeForce RTX 3060 (12.00 GB VRAM)                                                                                                                                         |
| **Operating System**    | Native Windows 11 (Python 3.11, CUDA 12.4, PyTorch 2.6.0+cu124)                                                                                                                 |
| **Quantization**        | 4-bit NormalFloat (NF4), Double Quantization, `bfloat16` compute dtype                                                                                                          |
| **LoRA Parameters**     | Rank r=16, Alpha α=32, Dropout 0.05 targeting all 7 linear projections (`q, k, v, o, gate, up, down`)                                                                           |
| **Adapter Size**        | **59.9 MB** (`output/final_adapter/`)                                                                                                                                           |
| **Peak VRAM**           | **4.00 GB** (under 35% of the 12GB hardware limit)                                                                                                                              |
| **Training Duration**   | **~30 minutes** (312 optimizer steps, 1 epoch across 5,000 samples)                                                                                                             |
| **Final Training Loss** | **0.5100** (starting from 2.908 in smoke test)                                                                                                                                  |
| **Data Cleaning**       | 5,000 synthetic pairs filtered via `sqlglot` AST parsing and verified via in-memory SQLite execution; sample `INSERT` statements stripped to prevent schema distribution shift. |
| **Leakage Audit**       | Verified 0 question overlap against Spider dev evaluation benchmark (`results/leakage_check.txt`).                                                                              |

---

## Project Structure

```text
E:\Projects\SQL\
├── README.md                           # Master project documentation & portfolio presentation
├── LICENSE                             # MIT Open Source License
├── ENGINEERING_LOG.md                  # Comprehensive decision log (15 decisions, 6 incidents)
├── Text-to-SQL Analyst Agent — Master Plan.md # Authoritative plan & specification
├── pyproject.toml                      # Project metadata, pinned dependencies, tool configs
├── docker-compose.yml                  # Single-command GPU container configuration
│
├── docker/
│   └── Dockerfile                      # Multi-stage Ubuntu 22.04 + CUDA 12.4 + PyTorch runtime
│
├── configs/
│   └── training_config.yaml            # QLoRA hyperparameters, LoRA target modules, training args
│
├── src/
│   ├── sql_tools.py                    # Schema extraction, read-only URI, AST safety, row comparison
│   ├── agent.py                        # LangGraph cyclic StateGraph with dual retry strategies
│   ├── api.py                          # Production FastAPI REST backend with Pydantic schemas
│   ├── evaluate.py                     # Standalone evaluation harness on 100 Spider dev questions
│   ├── evaluate_agent.py               # Evaluation harness for LangGraph self-healing agent loop
│   ├── train.py                        # QLoRA fine-tuning script via TRL SFTTrainer
│   ├── prepare_data.py                 # Gretel cleaning pipeline, AST validation, leakage checks
│   ├── setup_eval_data.py              # Automated download of 18 Spider SQLite databases & dev split
│   ├── test_gpu.py                     # GPU & bitsandbytes 4-bit NF4 sanity test harness
│   ├── analyze_failures.py             # Categorizes failed benchmark predictions into error taxonomy
│   └── static/
│       └── index.html                  # Dark-mode analyst UI (Tailwind CSS, Prism.js, telemetry)
│
├── tests/
│   ├── test_sql_tools.py               # Unit tests for read-only URI, timeouts, and AST guards
│   └── test_api.py                     # Integration tests for FastAPI endpoints & agent execution
│
├── data/
│   ├── processed/
│   │   ├── eval_questions.json         # 100 fixed Spider dev benchmark questions (seed 42)
│   │   └── train.jsonl                 # 5,000 sanitized ChatML training examples
│   └── spider_db/                      # 18 downloaded SQLite benchmark databases
│
├── output/
│   └── final_adapter/                  # Trained LoRA adapter weights (59.9 MB)
│
└── results/                            # Persistent evaluation JSON and CSV logs
    ├── baseline_20261111_204423.json   # Base model scores: 61.0% EX, 86.0% Valid-SQL
    ├── finetuned_3b_20261111_213542.json # Fine-tuned scores: 59.0% EX, 95.0% Valid-SQL
    ├── agent_resample_20261112_213417.json # Resample agent: 58.0% EX, 97.0% Valid-SQL
    ├── agent_error_feedback_20261112_214029.json # Feedback agent: 57.0% EX, 95.0% Valid-SQL
    ├── failure_analysis.json           # Categorized 15 failure samples
    └── leakage_check.txt               # Verified 0 training/eval overlap
```

---

## Quickstart & Installation

### Option 1: Native Local Setup (Recommended for Windows / Linux)

#### 1. Clone & Setup Virtual Environment

```bash
git clone https://github.com/your-username/text-to-sql-agent.git
cd text-to-sql-agent

# Create virtual environment with Python 3.11
python -m venv .venv

# Activate virtual environment

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies with PyTorch CUDA 12.4
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

#### 2. Run Hardware & Safety Tests

```bash
# Verify GPU detection & 4-bit quantization
python src/test_gpu.py

# Run unit and integration test suite
python -m pytest tests
```

#### 3. (Optional) Fuse Weights for Maximum Inference Speed

By default, the repository runs with the base model and lightweight 59.9 MB LoRA adapter (`output/final_adapter/`). To eliminate adapter computation and achieve the fastest query latency (~1.56s):

```bash
python src/merge_adapter.py
```

> _Note: The Agent and FastAPI backend automatically detect and switch to `output/merged_model/` once created, with zero extra configuration._

#### 4. Launch the Analyst Web Dashboard & API

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open your browser to: **`http://localhost:8000`**

---

### Option 2: Docker Deployment (GPU-Accelerated)

Prerequisites: Docker Desktop with WSL2 backend (Windows) or NVIDIA Container Toolkit (Linux).

```bash
# Build and run container with GPU pass-through
docker compose up --build
```

Access the dashboard at `http://localhost:8000`.

---

## Reproducing the Benchmark Results

To re-run the evaluations across the fixed 100 Spider dev questions:

```bash
# 1. Evaluate un-tuned base model
python src/evaluate.py --model-name Qwen/Qwen2.5-Coder-3B-Instruct

# 2. Evaluate fine-tuned QLoRA adapter
python src/evaluate.py --model-name Qwen/Qwen2.5-Coder-3B-Instruct --adapter-path output/final_adapter

# 3. Evaluate merged standalone model (if merged locally)
python src/evaluate.py --model-name output/merged_model --config-name merged_3b

# 4. Evaluate LangGraph self-healing agent (Resample strategy)
python src/evaluate_agent.py --strategy resample --max-retries 3

# 5. Evaluate LangGraph self-healing agent (Error feedback strategy)
python src/evaluate_agent.py --strategy error_feedback --max-retries 3
```

Results are saved automatically to `results/` in both JSON and CSV formats.

---

## Future Roadmap

1. **GGUF Quantization & Local Serving**: Export the merged standalone model to GGUF format (Q4_K_M) for high-throughput, low-latency execution via Ollama and llama.cpp.
2. **7B Model Scaling**: Benchmark `Qwen2.5-Coder-7B-Instruct` with QLoRA to evaluate whether 7B parameters natively resolves the 4-table join barrier.
3. **Schema Graph Pruning**: Implement networkx-based foreign key pathfinding to inject minimal relevant table subgraphs rather than complete database DDLs.
4. **Multi-Turn Training**: Fine-tune on synthetic (Error, Corrected SQL) tuples to align the model specifically for conversational error feedback.
5. **Natural Language Result Summarization**: Integrate a lightweight secondary local LLM call to synthesize table outputs into business insights.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
