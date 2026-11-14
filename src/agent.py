from pathlib import Path
from typing import TypedDict
import torch
from peft import PeftModel
from langgraph.graph import StateGraph, START, END
from rich.console import Console

from evaluate import Evaluator, SYSTEM_PROMPT
from evaluate import build_chatml_prompt
import sql_tools

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SQLGenerator:

    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        max_new_tokens: int = 128,
    ):
        self.max_new_tokens = max_new_tokens

        # Auto-detect standalone merged model if available
        if model_name is None:
            merged_dir = PROJECT_ROOT / "output" / "merged_model"
            adapter_dir = PROJECT_ROOT / "output" / "final_adapter"

            if (merged_dir / "config.json").exists() and (merged_dir / "model.safetensors").exists():
                console.print(f"[bold green]Auto-detected Merged Model:[/bold green] {merged_dir} (Fused standalone weights)")
                model_name = str(merged_dir)
                adapter_path = None
            else:
                model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
                if adapter_path is None and adapter_dir.exists():
                    adapter_path = str(adapter_dir)

        self.evaluator = Evaluator(base_model_name=model_name, adapter_path=adapter_path)
        self.evaluator.load_model()
        self.model = self.evaluator.model
        self.tokenizer = self.evaluator.tokenizer
   

    def generate(self, schema_ddl:str, question:str, attempts:list[dict], strategy:str) -> str:
          
        if not attempts:
            sql,_ = self.evaluator.generate_sql(schema_ddl, question, max_new_tokens=self.max_new_tokens)
            return sql
        
        if strategy == "resample":
            return self.generate_resample(schema_ddl, question)

        elif strategy == "error_feedback":
            return self.generate_with_feedback(schema_ddl, question, attempts)

        else:
            sql,_ = self.evaluator.generate_sql(schema_ddl, question, max_new_tokens=self.max_new_tokens)
            return sql

    def generate_resample(self, schema_ddl:str, question:str) -> str:

        prompt = build_chatml_prompt(self.tokenizer, schema_ddl, question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens = self.max_new_tokens,
                do_sample = True,
                temperature = 0.7,
                top_p = 0.9,
                pad_token_id = self.tokenizer.pad_token_id,
                eos_token_id = self.tokenizer.eos_token_id,
            )
        
        new_tokens = output_tokens[0][input_len:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return sql_tools.extract_raw_sql(raw_output)

            
    def generate_with_feedback(self, schema_ddl:str , question:str , attempts:list[dict]) -> str:

        last_attempt = attempts[-1]
        last_sql = last_attempt["sql"]
        last_error = last_attempt["error"] or "Syntax or execution error"

        user_turn_1 = f"Database schema:\n{schema_ddl.strip()}\n\nQuestion: {question.strip()}"
        user_turn_2 = f"That query failed with error: {last_error}. Please provide a corrected SQLite query in a ```sql code block and nothing else."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_turn_1},
            {"role": "assistant", "content": f"```sql\n{last_sql}\n```"},
            {"role": "user", "content": user_turn_2},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy correction
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_tokens[0][input_len:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return sql_tools.extract_raw_sql(raw_output)

#Global instance of generator (Makes sure only 1 instance runs at a time)
generator = None

def get_generator():

    global generator
    if generator is None:
        generator = SQLGenerator()

    return generator

class SQLAgentState(TypedDict):

    question: str
    db_path: str
    schema_ddl: str
    sql: str
    columns: list[str]
    rows: list[tuple]
    is_truncated: bool
    error_message: str | None
    attempts: list[dict]
    retry_count: int
    max_retries: int
    strategy: str
    status: str

def setup_node(state: SQLAgentState):

    schema = state.get("schema_ddl") or sql_tools.get_schema(state["db_path"])
    
    return {
        "schema_ddl" : schema,
        "attempts" : [],
        "retry_count": 0,
        "status": "running",
    }

def generate_sql_node(state:SQLAgentState):

    generator = get_generator()
    
    sql = generator.generate(state["schema_ddl"], state["question"], state["attempts"], state["strategy"])

    return{
        "sql": sql,
    }

def validate_ast_node(state: SQLAgentState):

    is_safe, error = sql_tools.is_safe_select(state["sql"])

    return{
        "error_message" : error if not is_safe else None
    }

def execute_query_node(state:SQLAgentState):

    conn = None
    try:
        conn = sql_tools.open_readonly(state["db_path"])
        cols, rows, is_truncated = sql_tools.run_query(conn, state["sql"])
        return{
            "columns": cols,
            "rows": rows,
            "is_truncated": is_truncated,
            "error_message": None,
            "status": "ok",
            "attempts": state["attempts"] + [{
                "attempt": state["retry_count"] + 1,
                "sql": state["sql"],
                "error": None
            }]
        }
    except Exception as e:
        return{
            "error_message": f"SQLite Error: {e}",
            "status": "failed",
        }
    finally:
        if conn:
            conn.close()
    

def reflect_node(state:SQLAgentState):
    new_retry_count = state["retry_count"] + 1
    new_attempt = {
        "attempt": new_retry_count,
        "sql": state["sql"],
        "error": state["error_message"],
    }

    return {
        "attempts": state["attempts"] + [new_attempt],
        "retry_count": new_retry_count,
        "status": "failed" if new_retry_count >= state.get("max_retries", 3) else "running",
    }

#---------------------------
# LANGGRAPH IMPLEMENTATION
#---------------------------

def route_after_ast(state: SQLAgentState) -> str:
    if state.get("error_message"):
        return "reflect"
    return "execute_query"

def route_after_exec(state: SQLAgentState) -> str:
    if state.get("status") == "ok":
        return END
    return "reflect"

def route_after_reflect(state: SQLAgentState) -> str:
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    if retry_count < max_retries:
        return "generate_sql"
    return END

def build_agent_graph():

    builder = StateGraph(SQLAgentState)

    builder.add_node("setup", setup_node)
    builder.add_node("generate_sql", generate_sql_node)
    builder.add_node("validate_sql", validate_ast_node)
    builder.add_node("execute_query", execute_query_node)
    builder.add_node("reflect", reflect_node)

    builder.add_edge(START, "setup")
    builder.add_edge("setup", "generate_sql")
    builder.add_edge("generate_sql", "validate_sql")
    
    builder.add_conditional_edges("validate_sql", route_after_ast)
    builder.add_conditional_edges("execute_query", route_after_exec)
    builder.add_conditional_edges("reflect", route_after_reflect)

    return builder.compile()

def run_agent(question: str, db_path: str, strategy: str = "resample", max_retries: int = 3) -> dict:
    """Convenience function to run the LangGraph agent programmatically."""
    app = build_agent_graph()
    initial_state = {
        "question": question,
        "db_path": db_path,
        "schema_ddl": "",
        "sql": "",
        "columns": [],
        "rows": [],
        "is_truncated": False,
        "error_message": None,
        "attempts": [],
        "retry_count": 0,
        "max_retries": max_retries,
        "strategy": strategy,
        "status": "pending",
    }
    return app.invoke(initial_state)


def main():
    import argparse
    from rich import print
    from rich.table import Table

    parser = argparse.ArgumentParser(description="Text-to-SQL LangGraph Analyst Agent")
    parser.add_argument("--db", type=str, required=True, help="Path to SQLite database")
    parser.add_argument("--question", type=str, required=True, help="User question")
    parser.add_argument("--strategy", choices=["resample", "error_feedback"], default="resample")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    app = build_agent_graph()

    initial_state = {
        "question": args.question,
        "db_path": args.db,
        "schema_ddl": "",
        "sql": "",
        "columns": [],
        "rows": [],
        "is_truncated": False,
        "error_message": None,
        "attempts": [],
        "retry_count": 0,
        "max_retries": args.max_retries,
        "strategy": args.strategy,
        "status": "pending",
    }

    result = app.invoke(initial_state)

    print(f"\n[bold green]Final Status: {result['status'].upper()}[/bold green]")
    print(f"[bold cyan]Final SQL:[/bold cyan]\n{result['sql']}\n")
    print(f"Attempts needed: {len(result['attempts'])}")

    # Print attempt log
    for att in result["attempts"]:
        print(f"  - Attempt {att['attempt']}: Error: {att['error']}")

    # Print table if rows exist
    if result["rows"]:
        table = Table(title="Query Result")
        for col in result["columns"]:
            table.add_column(col, style="cyan")
        for row in result["rows"][:20]:
            table.add_row(*[str(val) for val in row])
        print(table)

if __name__ == "__main__":
    main()
    