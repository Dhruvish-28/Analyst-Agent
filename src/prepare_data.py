"""Data preparation pipeline: downloads, filters, and formats training data into ChatML.

Complies strictly with Text-to-SQL Analyst Agent — Master Plan:
1. Filters examples by sqlglot parsing (SQLite dialect) AND by running on in-memory SQLite.
2. Applies the standard ChatML prompt template matching evaluation and agent.
3. Produces ~5,000 high-quality training pairs in data/processed/train.jsonl (max 1024 tokens).
4. Runs strict leakage check against data/processed/eval_questions.json and writes results to results/leakage_check.txt.
"""

import argparse
import json
from pathlib import Path
import random
import re
import sqlite3
import sys

from datasets import load_dataset
from rich.console import Console
from rich.table import Table
import sqlglot
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
import sql_tools

console = Console()

SYSTEM_PROMPT = (
    "You are an expert SQL generator. Given a database schema and a question, "
    "return one valid SQLite query in a ```sql code block and nothing else."
)


def clean_schema_ddl(raw_schema: str) -> str:
    """Cleans schema context so it ONLY contains CREATE TABLE statements.

    Strips out all INSERT INTO rows so that training prompts match evaluation prompts
    (which have only CREATE TABLE statements and no sample rows).
    """
    try:
        stmts = sqlglot.parse(raw_schema, read="sqlite")
        creates = [s.sql(dialect="sqlite") + ";" for s in stmts if s and isinstance(s, sqlglot.exp.Create)]
        if creates:
            return "\n\n".join(creates)
    except Exception:
        pass

    # Fallback if sqlglot parse had dialect quirks
    cleaned = []
    for stmt in raw_schema.split(";"):
        stripped = stmt.strip()
        if stripped.upper().startswith("CREATE TABLE"):
            cleaned.append(stripped + ";")
    return "\n\n".join(cleaned)


def verify_sqlite_execution(schema_ddl: str, sql: str) -> bool:
    """Builds an in-memory SQLite database from schema DDL and executes the SQL query.

    Returns True only if schema builds and query executes without database errors.
    """
    clean_sql = sql_tools.extract_raw_sql(sql)
    if not clean_sql:
        return False

    conn = None
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        # 1. Execute schema creation script
        cursor.executescript(schema_ddl)
        conn.commit()

        # 2. Execute query
        cursor.execute(clean_sql)
        return True
    except Exception:
        return False
    finally:
        if conn:
            conn.close()


def format_chatml_record(tokenizer, schema_ddl: str, question: str, sql: str) -> str:
    """Formats schema, question, and SQL query into standard ChatML string."""
    clean_sql = sql_tools.extract_raw_sql(sql)
    assistant_content = f"```sql\n{clean_sql}\n```"
    user_content = f"Database schema:\n{schema_ddl.strip()}\n\nQuestion: {question.strip()}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def run_leakage_check(train_records: list[dict], eval_questions_path: Path, output_file: Path) -> dict:
    """Verifies that no evaluation questions appear in the training dataset."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not eval_questions_path.exists():
        console.print(f"[yellow]Warning: {eval_questions_path} not found. Skipping leakage check.[/yellow]")
        return {"status": "skipped", "leakage_count": 0}

    with open(eval_questions_path, "r", encoding="utf-8") as f:
        eval_samples = json.load(f)

    # Normalize question texts: lowercase and remove punctuation
    def _normalize_text(txt: str) -> str:
        return re.sub(r"[^\w\s]", "", txt.lower().strip())

    eval_questions_norm = {_normalize_text(s["question"]): s for s in eval_samples}
    eval_db_ids = {s["db_id"].lower() for s in eval_samples if "db_id" in s}

    leaked_questions = []

    for idx, train_item in enumerate(train_records):
        norm_q = _normalize_text(train_item.get("question", ""))
        if norm_q in eval_questions_norm:
            leaked_questions.append({
                "train_idx": idx,
                "question": train_item["question"],
                "matched_eval_id": eval_questions_norm[norm_q].get("eval_id"),
            })

    is_clean = len(leaked_questions) == 0

    report_text = (
        f"=== LEAKAGE CHECK REPORT ===\n"
        f"Date: {Path(output_file).name}\n"
        f"Total Training Samples Checked: {len(train_records)}\n"
        f"Total Eval Questions Checked: {len(eval_samples)}\n"
        f"Direct Leaked Questions: {len(leaked_questions)}\n"
        f"Leakage Status: {'PASSED (Zero Overlap)' if is_clean else 'FAILED (Overlap Detected)'}\n"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)
        if leaked_questions:
            f.write("\nLeaked Details:\n")
            for item in leaked_questions:
                f.write(f"- Train #{item['train_idx']}: {item['question']}\n")

    console.print(f"[bold {'green' if is_clean else 'red'}]{report_text}[/bold {'green' if is_clean else 'red'}]")
    return {
        "status": "passed" if is_clean else "failed",
        "leakage_count": len(leaked_questions),
        "total_train": len(train_records),
    }


def prepare_training_dataset(
    target_count: int = 5000,
    max_seq_length: int = 1024,
    output_path: Path = Path("data/processed/train.jsonl"),
    eval_questions_path: Path = Path("data/processed/eval_questions.json"),
    leakage_report_path: Path = Path("results/leakage_check.txt"),
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
) -> list[dict]:
    """Ingests, cleans, formats, and validates the training dataset."""
    console.print(f"[bold cyan]Loading Tokenizer:[/bold cyan] {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    console.print(f"[bold cyan]Loading Dataset:[/bold cyan] gretelai/synthetic_text_to_sql (split: train)")
    dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")

    console.print(f"Total raw examples available: {len(dataset)}")
    console.print(f"Filtering examples by AST syntax, SQLite execution, and token budget (max {max_seq_length})...")

    valid_samples: list[dict] = []
    total_scanned = 0

    for row in dataset:
        total_scanned += 1
        raw_schema = row.get("sql_context", "") or ""
        question = row.get("sql_prompt", "") or ""
        sql = row.get("sql", "") or ""

        if not raw_schema.strip() or not question.strip() or not sql.strip():
            continue

        schema_ddl = clean_schema_ddl(raw_schema)

        # 1. AST syntax check via sqlglot (sqlite dialect)
        is_valid_ast, _ = sql_tools.is_safe_select(sql)
        if not is_valid_ast:
            continue

        # 2. In-memory SQLite execution check
        if not verify_sqlite_execution(schema_ddl, sql):
            continue

        # 3. Format into ChatML and measure token length
        formatted_chatml = format_chatml_record(tokenizer, schema_ddl, question, sql)
        tokens = tokenizer.encode(formatted_chatml)
        if len(tokens) > max_seq_length:
            continue

        valid_samples.append({
            "question": question,
            "schema_ddl": schema_ddl,
            "sql_query": sql_tools.extract_raw_sql(sql),
            "text": formatted_chatml,
            "token_count": len(tokens),
        })

        if len(valid_samples) % 500 == 0:
            console.print(f"  Processed {len(valid_samples)} / {target_count} target samples...")

        if len(valid_samples) >= target_count:
            break

    console.print(f"[green]Filtering complete: selected {len(valid_samples)} valid samples (scanned {total_scanned}).[/green]")

    # Run leakage check before saving
    run_leakage_check(valid_samples, eval_questions_path, leakage_report_path)

    # Save to train.jsonl
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in valid_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    console.print(f"[bold green]Saved {len(valid_samples)} training pairs to {output_path}[/bold green]")
    return valid_samples


def main():
    parser = argparse.ArgumentParser(description="Prepare NL-to-SQL ChatML Training Dataset")
    parser.add_argument("--count", type=int, default=5000, help="Target number of training samples (default: 5000)")
    parser.add_argument("--max-len", type=int, default=1024, help="Maximum sequence token length (default: 1024)")
    parser.add_argument("--output", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--eval-file", type=str, default="data/processed/eval_questions.json")
    parser.add_argument("--leakage-file", type=str, default="results/leakage_check.txt")
    parser.add_argument("--inspect", action="store_true", help="Print 3 random formatted samples")

    args = parser.parse_args()

    samples = prepare_training_dataset(
        target_count=args.count,
        max_seq_length=args.max_len,
        output_path=Path(args.output),
        eval_questions_path=Path(args.eval_file),
        leakage_report_path=Path(args.leakage_file),
    )

    if args.inspect and samples:
        console.print("\n[bold cyan]=== Sample Inspection (ChatML Format) ===[/bold cyan]\n")
        inspect_samples = random.sample(samples, min(3, len(samples)))
        for i, s in enumerate(inspect_samples, 1):
            console.print(f"[bold yellow]--- Sample {i} ({s['token_count']} tokens) ---[/bold yellow]")
            console.print(s["text"])
            console.print("\n")


if __name__ == "__main__":
    main()

