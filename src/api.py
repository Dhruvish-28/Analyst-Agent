"""FastAPI Backend for Text-to-SQL Analyst Agent.

Provides production-grade REST API endpoints for:
- System and GPU health check (/health, /api/v1/health)
- SQLite database discovery and schema inspection (/databases, /api/v1/databases)
- Natural language query execution via LangGraph self-healing agent (/query, /api/v1/query)
"""

from contextlib import asynccontextmanager
from pathlib import Path
import sys
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch

# Ensure src/ is in sys.path so sibling imports work from any working directory
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent import run_agent, get_generator
import sql_tools

# Base directory for Spider SQLite databases
DB_BASE_DIR = PROJECT_ROOT / "data" / "spider_db"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI server."""
    print("[API] Text-to-SQL Analyst Agent API starting up...")
    try:
        print("[API] Preloading Text-to-SQL model into GPU VRAM (warming up)...")
        get_generator()
        print("[API] Model preloaded and ready in VRAM. First query will be instant!")
    except Exception as e:
        print(f"[API] Warning: Model preloading failed, will lazy-load on first query: {e}")
    yield
    print("[API] Shutting down Text-to-SQL API server.")


app = FastAPI(
    title="Text-to-SQL Analyst Agent API",
    description="REST API powered by Qwen2.5-Coder-3B QLoRA and LangGraph Self-Healing Agent",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for browser frontends and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and serve Web UI at root (/)
STATIC_DIR = SRC_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serves the analyst web dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Text-to-SQL Analyst Agent API is running. UI not found."}


# ============================================================================
# Pydantic Schemas
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    cuda_available: bool
    device_name: Optional[str] = None
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    model_loaded: bool


class DatabaseItem(BaseModel):
    db_id: str
    file_path: str
    size_mb: float
    tables: list[str]
    schema_ddl: str


class DatabaseListResponse(BaseModel):
    total: int
    databases: list[DatabaseItem]


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question to query the database")
    db_id: str = Field(..., description="Target SQLite database ID (e.g. 'singer', 'concert_singer')")
    strategy: str = Field(
        default="resample",
        description="Retry strategy for self-healing: 'resample' or 'error_feedback'"
    )
    max_retries: int = Field(default=3, ge=1, le=5, description="Maximum retry attempts")


class AttemptDetail(BaseModel):
    attempt: int
    sql: str
    error: Optional[str] = None


class QueryResponse(BaseModel):
    status: str
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    is_truncated: bool
    attempts: list[AttemptDetail]
    attempts_count: int
    healed: bool
    latency_seconds: float
    error_message: Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def check_health():
    """Returns GPU availability, VRAM memory usage, and model load status."""
    cuda_available = torch.cuda.is_available()
    device_name = None
    used_gb = None
    total_gb = None

    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_gb = round((total_bytes - free_bytes) / (1024 ** 3), 2)
            total_gb = round(total_bytes / (1024 ** 3), 2)
        except Exception:
            pass

    # Check if generator is initialized in agent.py
    import agent
    model_loaded = agent.generator is not None

    return HealthResponse(
        status="healthy",
        cuda_available=cuda_available,
        device_name=device_name,
        vram_used_gb=used_gb,
        vram_total_gb=total_gb,
        model_loaded=model_loaded,
    )


@app.get("/databases", response_model=DatabaseListResponse, tags=["Databases"])
@app.get("/api/v1/databases", response_model=DatabaseListResponse, tags=["Databases"])
def list_databases():
    """Lists all available SQLite databases in the benchmark repository with table lists and schemas."""
    if not DB_BASE_DIR.exists():
        raise HTTPException(status_code=500, detail=f"Database folder not found: {DB_BASE_DIR}")

    db_items: list[DatabaseItem] = []

    for sqlfile in sorted(DB_BASE_DIR.rglob("*.sqlite")):
        db_id = sqlfile.parent.name
        size_bytes = sqlfile.stat().st_size
        size_mb = round(size_bytes / (1024 * 1024), 2)

        tables: list[str] = []
        schema_ddl = ""
        try:
            conn = sql_tools.open_readonly(sqlfile)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
            )
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            schema_ddl = sql_tools.get_schema(sqlfile)
        except Exception as e:
            schema_ddl = f"Error extracting schema: {e}"

        db_items.append(
            DatabaseItem(
                db_id=db_id,
                file_path=str(sqlfile).replace("\\", "/"),
                size_mb=size_mb,
                tables=tables,
                schema_ddl=schema_ddl,
            )
        )

    return DatabaseListResponse(total=len(db_items), databases=db_items)


@app.get("/api/v1/databases/{db_id}/schema", tags=["Databases"])
def get_database_schema(db_id: str):
    """Retrieves schema DDL and table list for a specific database."""
    target_path = DB_BASE_DIR / db_id / f"{db_id}.sqlite"
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Database '{db_id}' not found.")

    schema = sql_tools.get_schema(target_path)
    return {"db_id": db_id, "schema_ddl": schema}


@app.post("/query", response_model=QueryResponse, tags=["Agent"])
@app.post("/api/v1/query", response_model=QueryResponse, tags=["Agent"])
def execute_query(req: QueryRequest):
    """Executes a user question against a database using the LangGraph self-healing agent."""
    target_path = DB_BASE_DIR / req.db_id / f"{req.db_id}.sqlite"
    if not target_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Database '{req.db_id}' not found at {target_path}. Use GET /api/v1/databases to list available databases."
        )

    if req.strategy not in ["resample", "error_feedback"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid strategy. Allowed values: 'resample', 'error_feedback'."
        )

    start_time = time.perf_counter()
    result = run_agent(
        question=req.question,
        db_path=str(target_path),
        strategy=req.strategy,
        max_retries=req.max_retries,
    )
    latency = time.perf_counter() - start_time

    # Format attempt traces
    raw_attempts = result.get("attempts", [])
    formatted_attempts = [
        AttemptDetail(
            attempt=att.get("attempt", idx + 1),
            sql=att.get("sql", ""),
            error=att.get("error"),
        )
        for idx, att in enumerate(raw_attempts)
    ]

    status = result.get("status", "failed")
    attempts_count = len(formatted_attempts)
    is_healed = bool(status == "ok" and attempts_count > 1)

    # Convert tuple rows to JSON-serializable lists
    raw_rows = result.get("rows", [])
    formatted_rows = [list(r) for r in raw_rows]

    return QueryResponse(
        status=status,
        sql=result.get("sql", ""),
        columns=result.get("columns", []),
        rows=formatted_rows,
        is_truncated=result.get("is_truncated", False),
        attempts=formatted_attempts,
        attempts_count=attempts_count,
        healed=is_healed,
        latency_seconds=round(latency, 3),
        error_message=result.get("error_message"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
