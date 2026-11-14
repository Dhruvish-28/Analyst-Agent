"""Unit and Integration tests for FastAPI endpoints in src/api.py.

Tests endpoint logic, schema validation, and LangGraph agent execution.
"""

import sys
from pathlib import Path

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import api


def test_api_endpoints():
    print("--- 1. Testing check_health() ---")
    health = api.check_health()
    print("Health Status:", health.status)
    print("Device Name:", health.device_name)
    print("CUDA Available:", health.cuda_available)
    print("VRAM (Used/Total):", f"{health.vram_used_gb} GB / {health.vram_total_gb} GB")
    assert health.status == "healthy"
    assert health.cuda_available is True

    print("\n--- 2. Testing list_databases() ---")
    db_list = api.list_databases()
    print("Total Databases Discovered:", db_list.total)
    assert db_list.total == 18
    assert len(db_list.databases) == 18
    first_db = db_list.databases[0]
    print(f"Sample DB: {first_db.db_id} | Tables: {first_db.tables} | Size: {first_db.size_mb} MB")

    print(f"\n--- 3. Testing get_database_schema('singer') ---")
    schema_res = api.get_database_schema("singer")
    print("Schema keys:", list(schema_res.keys()))
    assert schema_res["db_id"] == "singer"
    assert "CREATE TABLE" in schema_res["schema_ddl"].upper()

    print("\n--- 4. Testing execute_query() with Pydantic QueryRequest ---")
    req = api.QueryRequest(
        question="How many singers are there?",
        db_id="singer",
        strategy="resample",
        max_retries=3,
    )
    res = api.execute_query(req)
    print("Execution Status:", res.status)
    print("Generated SQL:", res.sql)
    print("Columns:", res.columns)
    print("Rows:", res.rows)
    print("Attempts Count:", res.attempts_count)
    print("Latency:", res.latency_seconds, "s")
    assert res.status == "ok"
    assert len(res.rows) > 0
    assert res.rows[0][0] == 6 or res.rows[0][0] == 8  # singer count in singer database

    print("\n[SUCCESS] All FastAPI endpoints and agent execution passed cleanly!")


if __name__ == "__main__":
    test_api_endpoints()
