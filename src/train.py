"""QLoRA fine-tuning script for Qwen2.5-Coder-3B-Instruct on an NVIDIA RTX 3060 12GB.

Complies strictly with Text-to-SQL Analyst Agent — Master Plan:
- 4-bit NF4 Quantization with double quantization and bfloat16 compute dtype.
- LoRA on all linear projection layers (r=16, alpha=32, dropout=0.05).
- SFTTrainer with gradient checkpointing and paged_adamw_8bit optimizer.
- Supports --smoke-test (2 training steps) to verify backpropagation and VRAM stability.
- Saves adapter weights to output/final_adapter/ and logs training metrics to results/.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from rich.table import Table
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer
import yaml

console = Console()


class LossLoggingCallback(TrainerCallback):
    """Logs step loss and peak VRAM to memory for JSON reporting."""

    def __init__(self):
        super().__init__()
        self.loss_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
            entry = {
                "step": state.global_step,
                "epoch": round(state.epoch or 0, 2),
                "loss": round(logs["loss"], 4),
                "learning_rate": logs.get("learning_rate"),
                "peak_vram_gb": round(peak_vram_gb, 2),
            }
            self.loss_history.append(entry)


def load_yaml_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def train(
    config_path: Path = Path("configs/qlora_qwen3b.yaml"),
    smoke_test: bool = False,
    override_epochs: int | None = None,
    train_file: str | None = None,
):
    start_wall_time = time.time()
    cfg = load_yaml_config(config_path)

    model_name = cfg.get("model", {}).get("base_model_name", "Qwen/Qwen2.5-Coder-3B-Instruct")
    training_cfg = cfg.get("training", {})
    lora_cfg = cfg.get("lora", {})

    output_dir = Path(training_cfg.get("output_dir", "output/checkpoints"))
    final_adapter_dir = Path(training_cfg.get("final_adapter_dir", "output/final_adapter"))
    results_dir = Path(cfg.get("eval", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_adapter_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = train_file or cfg.get("data", {}).get("train_file", "data/processed/train.jsonl")
    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"Training dataset not found at {dataset_path}. Run src/prepare_data.py first!"
        )

    console.print(f"\n[bold cyan]=== Starting Fine-Tuning Pipeline ===[/bold cyan]")
    console.print(f"Base Model: [bold green]{model_name}[/bold green]")
    console.print(f"Dataset: [bold green]{dataset_path}[/bold green]")
    console.print(f"Smoke Test: [bold yellow]{smoke_test}[/bold yellow]\n")

    # 1. Tokenizer
    console.print("[cyan]Loading Tokenizer...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.get("model", {}).get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=cfg.get("model", {}).get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 3. Base Model in 4-bit
    console.print("[cyan]Loading base model in 4-bit NF4...[/cyan]")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Prepare for k-bit training and gradient checkpointing
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
    )

    # 4. LoRA Config
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )

    # 5. Load Dataset
    console.print("[cyan]Loading training dataset...[/cyan]")
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

    if smoke_test:
        raw_dataset = raw_dataset.select(range(min(16, len(raw_dataset))))
        epochs = 1
        max_steps = 2
        save_steps = 2
        logging_steps = 1
        warmup_steps = 0
        console.print("[bold yellow]Smoke Test Active: Limited to 16 samples and 2 steps.[/bold yellow]")
    else:
        epochs = override_epochs or training_cfg.get("num_train_epochs", 1)
        max_steps = -1
        save_steps = training_cfg.get("save_steps", 200)
        logging_steps = training_cfg.get("logging_steps", 10)
        # Calculate ~3% warmup steps
        effective_batch_size = (
            training_cfg.get("per_device_train_batch_size", 2)
            * training_cfg.get("gradient_accumulation_steps", 8)
        )
        total_steps = (len(raw_dataset) // effective_batch_size) * epochs
        warmup_steps = max(1, int(total_steps * training_cfg.get("warmup_ratio", 0.03)))

    # 6. SFT Training Arguments
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field=cfg.get("data", {}).get("dataset_text_field", "text"),
        max_length=training_cfg.get("max_seq_length", 1024),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        logging_steps=logging_steps,
        save_strategy=training_cfg.get("save_strategy", "steps"),
        save_steps=save_steps,
        save_total_limit=training_cfg.get("save_total_limit", 2),
        seed=training_cfg.get("seed", 42),
        bf16=training_cfg.get("bf16", True),
        fp16=training_cfg.get("fp16", False),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 0),
        report_to="none",
    )

    loss_callback = LossLoggingCallback()

    trainer = SFTTrainer(
        model=model,
        train_dataset=raw_dataset,
        peft_config=peft_config,
        args=sft_config,
        processing_class=tokenizer,
        callbacks=[loss_callback],
    )

    # Reset max VRAM tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    console.print("\n[bold green]=== Commencing Training ===[/bold green]\n")
    train_result = trainer.train()

    # 7. Post-training statistics
    duration_minutes = (time.time() - start_wall_time) / 60.0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

    if not smoke_test:
        console.print(f"[bold green]Saving final LoRA adapter to:[/bold green] {final_adapter_dir}")
        trainer.model.save_pretrained(str(final_adapter_dir))
        tokenizer.save_pretrained(str(final_adapter_dir))

    # Save summary report in results/
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "smoke_test" if smoke_test else f"run_{epochs}ep"
    report_path = results_dir / f"training_{mode_tag}_{timestamp_str}.json"

    metrics_summary = {
        "mode": "smoke_test" if smoke_test else "full_training",
        "timestamp": datetime.now().isoformat(),
        "base_model": model_name,
        "epochs": epochs,
        "total_training_samples": len(raw_dataset),
        "duration_minutes": round(duration_minutes, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "train_loss": train_result.training_loss,
        "loss_history": loss_callback.loss_history,
        "final_adapter_path": str(final_adapter_dir) if not smoke_test else None,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # Print summary table
    table = Table(title="Training Run Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Status", "SUCCESS")
    table.add_row("Mode", mode_tag)
    table.add_row("Training Samples", str(len(raw_dataset)))
    table.add_row("Final Loss", f"{train_result.training_loss:.4f}")
    table.add_row("Elapsed Time", f"{duration_minutes:.2f} minutes")
    table.add_row("Peak VRAM Detected", f"{peak_vram_gb:.2f} GB")
    table.add_row("Saved Metrics", str(report_path))
    if not smoke_test:
        table.add_row("Adapter Saved", str(final_adapter_dir))

    console.print("\n")
    console.print(table)
    console.print("\n")

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning for Qwen 2.5 Coder 3B")
    parser.add_argument("--config", type=str, default="configs/qlora_qwen3b.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2 steps to verify backprop and VRAM")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--train-file", type=str, default=None, help="Override training JSONL path")
    args = parser.parse_args()

    train(
        config_path=Path(args.config),
        smoke_test=args.smoke_test,
        override_epochs=args.epochs,
        train_file=args.train_file,
    )


if __name__ == "__main__":
    main()

