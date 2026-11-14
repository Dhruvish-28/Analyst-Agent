"""Agent Evaluation Harness for the 100 fixed Spider dev benchmark questions.

Evaluates the LangGraph self-healing agent against the benchmark:
- Measures Execution Accuracy (EX), Valid-SQL Rate, Latency, Attempts per Query,
  and Self-Healing Rate (queries where attempt 1 failed but a retry recovered the correct answer).
- Supports both retry strategies: 'resample' and 'error_feedback'.
- Strictly adheres to Text-to-SQL Analyst Agent — Master Plan.
- Outputs timestamped JSON and CSV artifacts to results/.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

from rich.console import Console
from rich.table import Table

from agent import run_agent, get_generator
import sql_tools

console = Console()


def evaluate_agent_benchmark(
    strategy: str = "resample",
    max_retries: int = 3,
    eval_file: str = "data/processed/eval_questions.json",
    spider_db_dir: str = "data/spider_db",
    results_dir: str = "results",
    limit: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Runs benchmark evaluation using the LangGraph agent."""
    eval_path = Path(eval_file)
    db_base = Path(spider_db_dir)
    res_dir = Path(results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    with open(eval_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if limit is not None:
        questions = questions[:limit]

    console.print(f"[bold cyan]Pre-warming model and agent...[/bold cyan]")
    get_generator()  # Ensure model is preloaded into VRAM

    console.print(
        f"\n[bold green]Starting Agent Evaluation ({len(questions)} questions)[/bold green]"
    )
    console.print(f"Strategy: [bold magenta]{strategy}[/bold magenta] | Max retries: [bold]{max_retries}[/bold]\n")

    per_question_results = []
    correct_count = 0
    valid_count = 0
    healed_count = 0
    total_attempts = 0
    total_time = 0.0

    for idx, item in enumerate(questions, 1):
        q_id = item.get("eval_id", idx)
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["sql"]

        db_path = item.get("db_path")
        if not db_path or not Path(db_path).exists():
            db_path = str(db_base / db_id / f"{db_id}.sqlite")

        has_order_by = "order by" in gold_sql.lower()

        # 1. Execute Gold Query to get reference results
        gold_conn = None
        gold_rows = None
        gold_error = None
        try:
            gold_conn = sql_tools.open_readonly(db_path)
            _, gold_rows, _ = sql_tools.run_query(gold_conn, gold_sql)
        except Exception as e:
            gold_error = str(e)
        finally:
            if gold_conn:
                gold_conn.close()

        if gold_error:
            console.print(f"[yellow]Skipping question {q_id} ({db_id}): Reference SQL error: {gold_error}[/yellow]")
            continue

        # 2. Run LangGraph Agent
        start_t = time.perf_counter()
        agent_res = run_agent(
            question=question,
            db_path=db_path,
            strategy=strategy,
            max_retries=max_retries,
        )
        latency = time.perf_counter() - start_t

        status = agent_res.get("status", "failed")
        final_sql = agent_res.get("sql", "")
        attempts = agent_res.get("attempts", [])
        attempts_count = len(attempts)
        total_attempts += attempts_count
        total_time += latency

        is_valid = (status == "ok")
        if is_valid:
            valid_count += 1

        is_correct = False
        error_msg = agent_res.get("error_message")

        if is_valid:
            agent_rows = agent_res.get("rows", [])
            is_correct = sql_tools.compare_results(gold_rows, agent_rows, has_order_by=has_order_by)
            if is_correct:
                correct_count += 1
        else:
            if attempts and attempts[-1].get("error"):
                error_msg = attempts[-1]["error"]

        healed = bool(is_correct and attempts_count > 1)
        if healed:
            healed_count += 1

        per_question_results.append({
            "eval_id": q_id,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "final_sql": final_sql,
            "status": status,
            "attempts_count": attempts_count,
            "valid_sql": is_valid,
            "is_correct": is_correct,
            "healed": healed,
            "latency_seconds": round(latency, 3),
            "error": error_msg or "",
        })

        # Real-time console status
        mark = "[bold green]✓[/bold green]" if is_correct else "[bold red]✗[/bold red]"
        heal_tag = " [bold cyan][HEALED][/bold cyan]" if healed else ""
        retry_tag = f" ({attempts_count} att)" if attempts_count > 1 else ""
        console.print(
            f"[{idx:>3}/{len(questions)}] {mark} ID {q_id:>3} ({db_id}){retry_tag}{heal_tag}: {latency:.2f}s"
        )

    total_evaluated = len(per_question_results)
    accuracy = (correct_count / total_evaluated * 100) if total_evaluated > 0 else 0.0
    valid_rate = (valid_count / total_evaluated * 100) if total_evaluated > 0 else 0.0
    avg_latency = (total_time / total_evaluated) if total_evaluated > 0 else 0.0
    avg_attempts = (total_attempts / total_evaluated) if total_evaluated > 0 else 0.0

    questions_with_retries = sum(1 for r in per_question_results if r["attempts_count"] > 1)
    retry_recovery_rate = (healed_count / questions_with_retries * 100) if questions_with_retries > 0 else 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "configuration": f"agent_{strategy}",
        "strategy": strategy,
        "max_retries": max_retries,
        "timestamp": timestamp,
        "total_evaluated": total_evaluated,
        "execution_accuracy": round(accuracy, 2),
        "valid_sql_rate": round(valid_rate, 2),
        "avg_latency_seconds": round(avg_latency, 3),
        "avg_attempts_per_query": round(avg_attempts, 2),
        "questions_requiring_retry": questions_with_retries,
        "healed_by_retry_count": healed_count,
        "retry_recovery_rate": round(retry_recovery_rate, 2),
    }

    # Save JSON summary & full results
    json_path = res_dir / f"agent_{strategy}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": summary,
            "results": per_question_results,
        }, f, indent=2)

    # Save CSV per-question breakdown
    csv_path = res_dir / f"agent_{strategy}_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "eval_id", "db_id", "question", "gold_sql", "final_sql",
                "status", "attempts_count", "valid_sql", "is_correct",
                "healed", "latency_seconds", "error",
            ]
        )
        writer.writeheader()
        writer.writerows(per_question_results)

    # Display Rich summary table
    table = Table(title=f"LangGraph Agent Benchmark Results ({strategy.upper()})")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold green")

    table.add_row("Total Evaluated", str(total_evaluated))
    table.add_row("Execution Accuracy (EX)", f"{accuracy:.1f}% ({correct_count}/{total_evaluated})")
    table.add_row("Valid-SQL Rate", f"{valid_rate:.1f}% ({valid_count}/{total_evaluated})")
    table.add_row("Avg Latency per Question", f"{avg_latency:.3f} s")
    table.add_row("Avg Attempts per Question", f"{avg_attempts:.2f}")
    table.add_row("Queries Requiring Retry", f"{questions_with_retries}")
    table.add_row("Self-Healed by Retry", f"{healed_count} (Recovery: {retry_recovery_rate:.1f}%)")
    table.add_row("Results JSON", str(json_path))
    table.add_row("Results CSV", str(csv_path))

    console.print("\n")
    console.print(table)

    return summary, per_question_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LangGraph Agent on Spider Benchmark")
    parser.add_argument("--strategy", choices=["resample", "error_feedback"], default="resample")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--eval-file", type=str, default="data/processed/eval_questions.json")
    parser.add_argument("--spider-db", type=str, default="data/spider_db")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions for smoke testing")
    args = parser.parse_args()

    evaluate_agent_benchmark(
        strategy=args.strategy,
        max_retries=args.max_retries,
        eval_file=args.eval_file,
        spider_db_dir=args.spider_db,
        results_dir=args.results_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
