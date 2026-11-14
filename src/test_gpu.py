"""Hardware and 4-bit Quantization Sanity Verification Script.

Checks CUDA availability, GPU device properties (RTX 3060 ~12GB VRAM),
and verifies that bitsandbytes 4-bit NF4 quantization executes properly on the GPU.
"""

import sys
import torch
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def check_cuda() -> bool:
    """Verifies CUDA availability, device name, and VRAM capacity."""
    if not torch.cuda.is_available():
        console.print("[bold red]CUDA is not available on this system![/bold red]")
        return False

    device_name = torch.cuda.get_device_name(0)
    total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
    total_vram_gb = total_vram_bytes / (1024**3)
    cuda_version = torch.version.cuda or "Unknown"
    torch_version = torch.__version__

    table = Table(title="GPU & Environment Verification", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="dim", width=25)
    table.add_column("Detected Value", style="bold green")

    table.add_row("CUDA Available", "True")
    table.add_row("Device Name", device_name)
    table.add_row("Total VRAM", f"{total_vram_gb:.2f} GB")
    table.add_row("CUDA Version", cuda_version)
    table.add_row("PyTorch Version", torch_version)
    table.add_row("bitsandbytes Version", bnb.__version__)

    console.print(table)

    if total_vram_gb < 10.0:
        console.print(
            f"[yellow]Warning: Detected {total_vram_gb:.2f} GB VRAM. Target is ~12 GB (RTX 3060).[/yellow]"
        )
    return True


def check_4bit_quantization() -> bool:
    """Verifies 4-bit NF4 quantization using bitsandbytes and Transformers config."""
    console.print("\n[bold cyan]Testing 4-bit NF4 Quantization Kernel...[/bold cyan]")

    try:
        # 1. Verify project-standard BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        console.print(f"  [green]✓[/green] BitsAndBytesConfig initialized (NF4 + Double Quant, bfloat16)")

        # 2. Allocate a 4-bit Linear layer directly on CUDA
        in_features, out_features = 256, 128
        linear_4bit = bnb.nn.Linear4bit(
            in_features,
            out_features,
            bias=False,
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
        ).to("cuda")

        # 3. Perform a forward pass to execute CUDA quantization kernels
        dummy_input = torch.randn(2, in_features, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            output = linear_4bit(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        allocated_vram = torch.cuda.memory_allocated() / (1024**2)  # in MB
        console.print(
            f"  [green]✓[/green] 4-bit Linear forward pass successful: output shape {list(output.shape)}"
        )
        console.print(
            f"  [green]✓[/green] GPU VRAM allocated for 4-bit layer: {allocated_vram:.2f} MB"
        )
        return True

    except Exception as e:
        console.print(f"[bold red]4-bit quantization check failed: {e}[/bold red]")
        return False


def main():
    console.print(Panel.fit("[bold white]Antigravity NL-to-SQL — GPU & Quantization Sanity Test[/bold white]"))

    cuda_ok = check_cuda()
    if not cuda_ok:
        sys.exit(1)

    quant_ok = check_4bit_quantization()
    if not quant_ok:
        sys.exit(1)

    console.print(
        "\n[bold green] All checks passed! Hardware is ready for QLoRA fine-tuning and 4-bit inference.[/bold green]\n"
    )


if __name__ == "__main__":
    main()
