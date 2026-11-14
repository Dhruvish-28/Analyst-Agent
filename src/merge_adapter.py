"""Model Adapter Merging Script (Weight Fusion).

Fuses QLoRA low-rank adapter weights directly into the base model weights,
producing a standalone, single-graph model in bfloat16 precision.
Eliminates dual-branch computation and peft dependency during inference.
"""

import sys
import time
from pathlib import Path

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()


def merge_and_save(
    base_model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    adapter_path: str = "output/final_adapter",
    output_dir: str = "output/merged_model",
) -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]1. Loading Tokenizer:[/bold cyan] {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold cyan]2. Loading Base Model in bfloat16 on {device}:[/bold cyan] {base_model_name}")
    start_load = time.perf_counter()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    load_time = time.perf_counter() - start_load
    console.print(f"   Base model loaded in {load_time:.2f}s")

    console.print(f"[bold cyan]3. Attaching LoRA Adapter:[/bold cyan] {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    console.print("[bold yellow]4. Merging Adapter Weights into Base Model (merge_and_unload)...[/bold yellow]")
    start_merge = time.perf_counter()
    merged_model = peft_model.merge_and_unload()
    merge_time = time.perf_counter() - start_merge
    console.print(f"   Weights fused in {merge_time:.2f}s")

    console.print(f"[bold green]5. Saving Standalone Merged Model to:[/bold green] {out_path}")
    start_save = time.perf_counter()
    merged_model.save_pretrained(out_path, safe_serialization=True)
    tokenizer.save_pretrained(out_path)
    save_time = time.perf_counter() - start_save
    console.print(f"   Model saved in {save_time:.2f}s")

    # Calculate total size
    total_size_mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / (1024 * 1024)
    console.print(f"[bold green]Merge complete![/bold green] Total size: {total_size_mb:.2f} MB")

    return out_path


if __name__ == "__main__":
    merge_and_save()
