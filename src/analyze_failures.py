import csv
import json
from pathlib import Path

def analyze_failures():
    # Find latest finetuned_3b csv
    results_dir = Path("results")
    csv_files = sorted(results_dir.glob("finetuned_3b_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        print("No finetuned_3b csv found.")
        return

    csv_file = csv_files[0]
    print(f"Reading: {csv_file}")
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        failures = [r for r in reader if r["execution_match"] == "False"]

    print(f"Total failures out of 100: {len(failures)}\n")

    failure_records = []
    for i, fail in enumerate(failures[:15], 1):
        record = {
            "index": i,
            "eval_id": fail.get("eval_id"),
            "db_id": fail.get("db_id"),
            "question": fail.get("question"),
            "gold_sql": fail.get("gold_sql"),
            "pred_sql": fail.get("predicted_sql"),
            "valid_sql": fail.get("valid_sql"),
            "error_msg": fail.get("error_message"),
        }
        failure_records.append(record)
        print(f"[{i}] DB: {record['db_id']}")
        print(f"  Question: {record['question']}")
        print(f"  Gold:     {record['gold_sql']}")
        print(f"  Pred:     {record['pred_sql']}")
        print(f"  Valid:    {record['valid_sql']} | Error: {record['error_msg']}\n")

    # Save to results/failure_analysis.json
    out_file = results_dir / "failure_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(failure_records, f, indent=2)
    print(f"Saved failure analysis to {out_file}")

if __name__ == "__main__":
    analyze_failures()

