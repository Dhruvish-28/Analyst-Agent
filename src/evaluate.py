"""Evaluation harness for benchmark scoring against the 100 fixed Spider dev questions.

Complies strictly with Text-to-SQL Analyst Agent — Master Plan:
- Evaluates Execution Accuracy (EX), Valid-SQL Rate, and Latency (seconds per question).
- Uses standard ChatML prompt template matching training and agent loop.
- Uses greedy decoding (temperature=0, max_new_tokens=128, stop at end-of-turn).
- Saves results to results/ as timestamped JSON and CSV files without overwriting.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

from peft import PeftModel
from rich.console import Console
from rich.table import Table
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import sql_tools

console = Console()

SYSTEM_PROMPT = (
    "You are an expert SQLite analyst. Given a database schema and a question, "
    "carefully identify the primary entity and select the corresponding table. "
    "When multiple tables contain similar columns (such as cities vs countries), "
    "ensure the query targets the exact entity requested by the user. "
    "Return one valid SQLite query in a ```sql code block and nothing else."
)

def build_chatml_prompt(tokenizer, schema_ddl: str, question: str) -> str:
    """Builds the exact prompt using ChatML and the tokenizer's chat template."""
    user_content = f"Database schema:\n{schema_ddl.strip()}\n\nQuestion: {question.strip()}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class Evaluator:
    """Runs evaluation on the test set for base or fine-tuned models."""

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
        adapter_path: str | None = None,
        config_name: str | None = None,
    ):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.config_name = config_name or ("finetuned_3b" if adapter_path else "baseline")
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Loads the model in 4-bit NF4 precision and attaches adapter if specified."""
        console.print(f"[bold cyan]Loading Tokenizer:[/bold cyan] {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        console.print(f"[bold cyan]Loading Model in 4-bit NF4:[/bold cyan] {self.base_model_name}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        if self.adapter_path:
            console.print(f"[bold green]Attaching LoRA Adapter:[/bold green] {self.adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        else:
            console.print("[yellow]Running in Base Model mode (no adapter)[/yellow]")
            self.model = base_model

        self.model.eval()

    def generate_sql(self, schema_ddl: str, question: str, max_new_tokens: int = 128) -> tuple[str, float]:
        """Runs greedy generation and measures latency in seconds."""
        prompt = build_chatml_prompt(self.tokenizer, schema_ddl, question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        start_time = time.perf_counter()
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding (temperature=0)
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_sec = time.perf_counter() - start_time

        new_tokens = output_tokens[0][input_len:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        clean_sql = sql_tools.extract_raw_sql(raw_output)
        return clean_sql, latency_sec

    def evaluate(
        self,
        eval_questions_path: Path,
        max_samples: int | None = None,
        output_dir: Path = Path("results"),
    ) -> dict[str, Any]:
        """Runs evaluation over the fixed questions and produces JSON and CSV reports."""
        if not eval_questions_path.exists():
            raise FileNotFoundError(f"Evaluation questions file not found: {eval_questions_path}")

        with open(eval_questions_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        if max_samples:
            samples = samples[:max_samples]

        total = len(samples)
        console.print(f"\n[bold green]Starting Evaluation on {total} questions ({self.config_name})...[/bold green]\n")

        valid_sql_count = 0
        execution_match_count = 0
        latencies: list[float] = []
        rows_for_csv: list[dict[str, Any]] = []

        for idx, sample in enumerate(samples, start=1):
            q_id = sample.get("eval_id", idx)
            db_path = sample["db_path"]
            question = sample["question"]
            gold_sql = sample["sql"]
            schema = sample["schema"]

            # Generate SQL
            pred_sql, latency = self.generate_sql(schema, question)
            latencies.append(latency)

            # Evaluate against database
            is_valid_syntax, syntax_err = sql_tools.is_safe_select(pred_sql)
            exec_match = False
            error_msg = syntax_err

            if Path(db_path).exists():
                try:
                    conn = sql_tools.open_readonly(db_path)
                    try:
                        # 1. Run reference query
                        has_order_by = "order by" in gold_sql.lower()
                        _, gold_rows, _ = sql_tools.run_query(conn, gold_sql)

                        # 2. Run predicted query
                        try:
                            _, pred_rows, _ = sql_tools.run_query(conn, pred_sql)
                            valid_sql_count += 1
                            # 3. Compare result sets
                            exec_match = sql_tools.compare_results(
                                gold_rows, pred_rows, has_order_by=has_order_by
                            )
                        except Exception as e:
                            error_msg = f"Runtime Error: {e}"
                    finally:
                        conn.close()
                except Exception as e:
                    error_msg = f"Database Open Error: {e}"
            else:
                error_msg = f"Database not found at {db_path}"

            if exec_match:
                execution_match_count += 1

            status_symbol = "✓" if exec_match else "✗"
            status_color = "green" if exec_match else "red"
            console.print(
                f"[{idx}/{total}] [{status_color}]{status_symbol}[/{status_color}] "
                f"EX: {exec_match} | Valid: {error_msg is None} | Latency: {latency:.2f}s"
            )

            rows_for_csv.append({
                "eval_id": q_id,
                "db_id": sample.get("db_id", ""),
                "question": question,
                "gold_sql": gold_sql,
                "predicted_sql": pred_sql,
                "execution_match": exec_match,
                "valid_sql": error_msg is None,
                "error_message": error_msg or "",
                "latency_sec": round(latency, 3),
            })

        # Aggregate metrics
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        ex_accuracy = (execution_match_count / total) * 100.0 if total else 0.0
        valid_rate = (valid_sql_count / total) * 100.0 if total else 0.0

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        json_file = output_dir / f"{self.config_name}_{timestamp_str}.json"
        csv_file = output_dir / f"{self.config_name}_{timestamp_str}.csv"

        metrics = {
            "config_name": self.config_name,
            "base_model": self.base_model_name,
            "adapter_path": self.adapter_path,
            "timestamp": datetime.now().isoformat(),
            "total_questions": total,
            "execution_accuracy_pct": round(ex_accuracy, 2),
            "valid_sql_rate_pct": round(valid_rate, 2),
            "avg_seconds_per_question": round(avg_latency, 3),
            "json_file": str(json_file),
            "csv_file": str(csv_file),
        }

        # Save JSON
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Save CSV
        if rows_for_csv:
            fieldnames = list(rows_for_csv[0].keys())
            with open(csv_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_for_csv)

        # Display summary table
        table = Table(title=f"Evaluation Results: {self.config_name.upper()}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold green")

        table.add_row("Total Questions", str(total))
        table.add_row("Execution Accuracy (EX)", f"{metrics['execution_accuracy_pct']}%")
        table.add_row("Valid-SQL Rate", f"{metrics['valid_sql_rate_pct']}%")
        table.add_row("Avg Seconds / Question", f"{metrics['avg_seconds_per_question']} s")
        table.add_row("Saved JSON", str(json_file))
        table.add_row("Saved CSV", str(csv_file))

        console.print("\n")
        console.print(table)
        console.print("\n")

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base or LoRA Model on Spider Dev Benchmark")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to LoRA adapter (if evaluating fine-tuned)")
    parser.add_argument("--config-name", type=str, default=None, help="Name tag for output files (e.g. baseline)")
    parser.add_argument("--eval-file", type=str, default="data/processed/eval_questions.json")
    parser.add_argument("--sample", type=int, default=None, help="Run on a subset (e.g. --sample 5 for testing)")
    args = parser.parse_args()

    evaluator = Evaluator(
        base_model_name=args.model_name,
        adapter_path=args.adapter_path,
        config_name=args.config_name,
    )
    evaluator.load_model()
    evaluator.evaluate(
        eval_questions_path=Path(args.eval_file),
        max_samples=args.sample,
    )


if __name__ == "__main__":
    main()

