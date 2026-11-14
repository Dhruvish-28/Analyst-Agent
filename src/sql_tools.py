"""SQLite utility tools for schema extraction, read-only query execution,

AST safety validation, and result set comparison.
Complies strictly with Text-to-SQL Analyst Agent — Master Plan.
"""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


def extract_raw_sql(generated_text: str) -> str:
    """Extracts raw SQL from markdown code blocks or clean strings."""
    text = generated_text.strip()
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def open_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Opens a SQLite database connection strictly in read-only mode using a URI.

    Prevents any disk mutations or write locks by the model.
    """
    resolved_path = Path(db_path).resolve().as_posix()
    # SQLite URI with mode=ro enforces filesystem read-only access
    uri = f"file:{resolved_path}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=10.0)


def get_schema(db_path: str | Path) -> str:
    """Reads CREATE TABLE statements from sqlite_master, skipping internal SQLite tables.

    The model prompt receives only these CREATE TABLE statements.
    """
    conn = open_readonly(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type='table'
              AND name NOT LIKE 'sqlite_%'
              AND sql IS NOT NULL
            ORDER BY name;
            """
        )
        ddl_statements = [row[0].strip() for row in cursor.fetchall() if row[0]]
        # Clean trailing semicolons and join cleanly
        cleaned_ddls = []
        for ddl in ddl_statements:
            statement = ddl.rstrip(";") + ";"
            cleaned_ddls.append(statement)
        return "\n\n".join(cleaned_ddls)
    finally:
        conn.close()


def is_safe_select(sql: str) -> tuple[bool, str | None]:
    """AST guardrail using sqlglot (SQLite dialect).

    Accepts exactly one statement that is strictly a SELECT (including WITH and UNION).
    Rejects multi-statements, empty queries, or any mutating statements.
    """
    clean_sql = extract_raw_sql(sql)
    if not clean_sql:
        return False, "Query is empty."

    try:
        statements = sqlglot.parse(clean_sql, read="sqlite")
    except ParseError as e:
        return False, f"SQL syntax parsing error: {e}"
    except Exception as e:
        return False, f"SQL parsing error: {e}"

    # Filter out empty None statements if parser returned any
    statements = [s for s in statements if s is not None]

    if len(statements) == 0:
        return False, "No valid SQL statement parsed."
    if len(statements) > 1:
        return False, "Multiple SQL statements are not permitted."

    stmt = statements[0]

    # Validate that root statement is a SELECT or UNION or WITH
    if not isinstance(stmt, (exp.Select, exp.Union)):
        return False, f"Disallowed statement type: {type(stmt).__name__}. Only SELECT is permitted."

    # Verify no mutating operations are contained within the expression tree
    disallowed_nodes = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Alter,
        exp.Create,
        exp.Command,
    )
    for node in stmt.find_all(disallowed_nodes):
        return False, f"Disallowed mutating operation found: {type(node).__name__}."

    return True, None


def run_query(
    conn: sqlite3.Connection,
    sql: str,
    timeout_seconds: float = 10.0,
    max_rows: int = 200,
) -> tuple[list[str], list[tuple[Any, ...]], bool]:
    """Runs a query on an open read-only SQLite connection.

    Enforces a strict timeout using SQLite's progress handler and caps rows at max_rows.
    Returns: (column_names, rows, is_truncated)
    """
    clean_sql = extract_raw_sql(sql)
    start_time = time.time()

    def _timeout_handler() -> int:
        # Returning non-zero interrupts the query
        if time.time() - start_time > timeout_seconds:
            return 1
        return 0

    # Check progress every 1,000 SQLite VM opcodes
    conn.set_progress_handler(_timeout_handler, 1000)
    cursor = conn.cursor()

    try:
        cursor.execute(clean_sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []

        # Fetch up to max_rows + 1 to detect truncation
        fetched_rows = cursor.fetchmany(max_rows + 1)
        is_truncated = len(fetched_rows) > max_rows
        returned_rows = fetched_rows[:max_rows]

        return columns, returned_rows, is_truncated
    finally:
        # Reset progress handler so it doesn't affect future queries
        conn.set_progress_handler(None, 0)


def _normalize_cell(val: Any) -> Any:
    """Normalizes cell values for robust comparison:

    - Rounds floats to 4 decimal places
    - Normalizes case and whitespace on strings
    """
    if val is None:
        return None
    if isinstance(val, float):
        return round(val, 4)
    if isinstance(val, (int, bool)):
        return val
    if isinstance(val, str):
        return val.strip().lower()
    return str(val).strip().lower()


def _normalize_row(row: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(_normalize_cell(v) for v in row)


def compare_results(
    gold_rows: list[tuple[Any, ...]],
    pred_rows: list[tuple[Any, ...]],
    has_order_by: bool = False,
) -> bool:
    """Compares database result sets according to the Master Plan:

    Correct if the result rows match as a multiset, ignoring column names and,
    unless the reference has ORDER BY, ignoring row order.
    """
    if len(gold_rows) != len(pred_rows):
        return False

    norm_gold = [_normalize_row(r) for r in gold_rows]
    norm_pred = [_normalize_row(r) for r in pred_rows]

    if has_order_by:
        return norm_gold == norm_pred

    # Multiset comparison: count occurrences of each normalized row
    try:
        return Counter(norm_gold) == Counter(norm_pred)
    except TypeError:
        # Fallback if any unhashable elements remain: compare sorted string representations
        return sorted([str(r) for r in norm_gold]) == sorted([str(r) for r in norm_pred])

