import sqlite3
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, "src")
import sql_tools

def test_sql_tools():
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INT, name TEXT, salary REAL);")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 100.5), (2, 'Bob', 200.0);")
    conn.commit()
    conn.close()

    schema = sql_tools.get_schema(db_path)
    assert "CREATE TABLE users" in schema

    ro_conn = sql_tools.open_readonly(db_path)
    safe, err = sql_tools.is_safe_select("SELECT name, salary FROM users;")
    assert safe, f"Expected safe: {err}"

    cols, rows, trunc = sql_tools.run_query(ro_conn, "SELECT name, salary FROM users;")
    assert len(rows) == 2
    assert cols == ["name", "salary"]

    safe2, err2 = sql_tools.is_safe_select("DROP TABLE users;")
    assert not safe2, "DROP TABLE should not be safe"

    # Multiset comparison: column order aligned (name, salary), row order permuted
    gold = [("Alice", 100.5), ("Bob", 200.0)]
    pred = [("Bob", 200.00001), ("alice", 100.5)]
    assert sql_tools.compare_results(gold, pred, has_order_by=False), "Multiset comparison failed"

    # Order-by comparison
    assert not sql_tools.compare_results(gold, pred, has_order_by=True), "Ordered comparison should have failed on permuted rows"

    ro_conn.close()
    Path(db_path).unlink()
    print("All sql_tools unit tests passed successfully!")

if __name__ == "__main__":
    test_sql_tools()
