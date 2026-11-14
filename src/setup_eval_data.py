"""Spider Dev dataset downloader and evaluation question sampler.

Complies strictly with Text-to-SQL Analyst Agent — Master Plan:
- Downloads Spider dev questions (dev.json) and SQLite databases from Hugging Face hub (Chinastark/spider_datasets).
- Fixes random seed (42) and samples exactly 100 questions.
- Downloads SQLite database files into data/spider_db/<db_id>/<db_id>.sqlite.
- Extracts schemas for each sampled question from its real SQLite database using sql_tools.get_schema.
- Saves fixed test set to data/processed/eval_questions.json.
"""

import json
from pathlib import Path
import random
import sys

from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.table import Table

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent))
import sql_tools

console = Console()
REPO_ID = "Chinastark/spider_datasets"


def fetch_dev_json(raw_dir: Path) -> Path:
    """Downloads dev.json from Hugging Face repository."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dev_path = raw_dir / "spider_dev.json"

    if dev_path.exists():
        console.print(f"[green]Using existing Spider dev.json at {dev_path}[/green]")
        return dev_path

    console.print(f"[cyan]Downloading dev.json from {REPO_ID}...[/cyan]")
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename="dev.json",
        repo_type="dataset",
        local_dir=str(raw_dir),
    )
    downloaded_path = Path(downloaded)
    if downloaded_path != dev_path:
        downloaded_path.rename(dev_path)

    console.print(f"[green]Downloaded dev.json to {dev_path}[/green]")
    return dev_path


def ensure_database(db_id: str, spider_db_dir: Path) -> Path:
    """Downloads a specific Spider SQLite database if not already present."""
    target_db_file = spider_db_dir / db_id / f"{db_id}.sqlite"
    if target_db_file.exists():
        return target_db_file

    console.print(f"  Downloading SQLite database: [yellow]{db_id}[/yellow]...")
    hf_path = f"database/{db_id}/{db_id}.sqlite"
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=hf_path,
        repo_type="dataset",
        local_dir=str(spider_db_dir.parent),
    )
    # The file is downloaded to data/database/<db_id>/<db_id>.sqlite
    downloaded_file = Path(downloaded)
    if not target_db_file.exists() and downloaded_file.exists():
        target_db_file.parent.mkdir(parents=True, exist_ok=True)
        if downloaded_file != target_db_file:
            import shutil
            shutil.copy2(downloaded_file, target_db_file)

    return target_db_file


def create_eval_questions(
    dev_json_path: Path,
    spider_db_dir: Path,
    output_path: Path,
    num_samples: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Samples fixed evaluation questions, ensures their SQLite databases are downloaded,

    extracts their schemas, and saves them.
    """
    with open(dev_json_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    console.print(f"Total Spider dev questions: {len(dev_data)}")

    random.seed(seed)
    indices = list(range(len(dev_data)))
    random.shuffle(indices)

    selected_samples = []
    skipped = 0

    console.print(f"\n[cyan]Sampling {num_samples} questions and downloading databases...[/cyan]")

    for idx in indices:
        item = dev_data[idx]
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]

        try:
            db_file = ensure_database(db_id, spider_db_dir)
            if not db_file.exists():
                skipped += 1
                continue

            schema = sql_tools.get_schema(db_file)
            if not schema.strip():
                skipped += 1
                continue
        except Exception as e:
            console.print(f"[yellow]Warning: Could not prepare {db_id}: {e}[/yellow]")
            skipped += 1
            continue

        selected_samples.append({
            "eval_id": len(selected_samples) + 1,
            "spider_dev_idx": idx,
            "db_id": db_id,
            "db_path": db_file.resolve().as_posix(),
            "question": question,
            "sql": gold_sql,
            "schema": schema,
        })

        if len(selected_samples) >= num_samples:
            break

    if len(selected_samples) < num_samples:
        raise RuntimeError(f"Could only find {len(selected_samples)} valid samples (needed {num_samples}).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_samples, f, indent=2, ensure_ascii=False)

    console.print(f"[bold green]Successfully saved {len(selected_samples)} evaluation questions to {output_path}[/bold green]")
    return selected_samples


def main():
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    spider_db_dir = data_dir / "spider_db"
    eval_output = data_dir / "processed" / "eval_questions.json"

    console.print("[bold cyan]=== Setting Up Spider Dev Evaluation Benchmark ===[/bold cyan]\n")
    dev_json_path = fetch_dev_json(raw_dir)
    samples = create_eval_questions(dev_json_path, spider_db_dir, eval_output, num_samples=100, seed=42)

    unique_dbs = len(set(s["db_id"] for s in samples))
    table = Table(title="Evaluation Benchmark Summary (100 Fixed Questions)")
    table.add_column("Property", style="dim")
    table.add_column("Value", style="bold green")

    table.add_row("Total Questions", str(len(samples)))
    table.add_row("Unique Databases", str(unique_dbs))
    table.add_row("Random Seed", "42 (Fixed)")
    table.add_row("Output File", str(eval_output))

    console.print("\n")
    console.print(table)
    console.print("\n")


if __name__ == "__main__":
    main()
