"""Command-line interface for the SLM SMS toolkit."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="SLM SMS: Indian SMS Entity Extraction & Fine-Tuning CLI")
console = Console()


@app.command("generate")
def generate_cli(
    samples: int = typer.Option(
        2000, "--samples", "-n", help="Total number of samples to generate"
    ),
    output_dir: Path = typer.Option(
        Path("./data"), "--output-dir", "-o", help="Target output directory"
    ),
    val_split: float = typer.Option(0.15, "--val-split", help="Validation set split ratio"),
    test_split: float = typer.Option(0.05, "--test-split", help="Test set split ratio"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
):
    """Generate high-fidelity synthetic Indian SMS dataset."""
    from slm_sms.data.dataset import save_jsonl
    from slm_sms.data.generator import IndianSMSGenerator

    console.print(f"[bold green]Generating {samples} Indian SMS samples...[/bold green]")
    generator = IndianSMSGenerator(seed=seed)
    splits = generator.generate_dataset(
        n_samples=samples, val_split=val_split, test_split=test_split
    )

    train_path = save_jsonl(splits["train"], output_dir / "train.jsonl")
    val_path = save_jsonl(splits["validation"], output_dir / "val.jsonl")
    test_path = save_jsonl(splits["test"], output_dir / "test.jsonl")

    table = Table(title="Generated Dataset Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("File Path", style="green")

    table.add_row("Train", str(len(splits["train"])), str(train_path))
    table.add_row("Validation", str(len(splits["validation"])), str(val_path))
    table.add_row("Test", str(len(splits["test"])), str(test_path))

    console.print(table)
    console.print("\n[bold]Sample Generated SMS:[/bold]")
    sample_msg = splits["train"][0]["messages"]
    console.print(f"[yellow]User SMS:[/yellow] {sample_msg[1]['content']}")
    console.print(f"[green]Extracted JSON:[/green] {sample_msg[2]['content']}")


@app.command("train")
def train_cli(
    model_id: str = typer.Option(
        "Qwen/Qwen3.5-2B", "--model-id", "-m", help="Hugging Face Model ID"
    ),
    data_dir: Path = typer.Option(
        Path("./data"), "--data-dir", "-d", help="Directory with train.jsonl & val.jsonl"
    ),
    output_dir: Path = typer.Option(
        Path("./output/qwen-sms-lora"), "--output-dir", "-o", help="Adapter save directory"
    ),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Per-device train batch size"),
    grad_accum: int = typer.Option(4, "--grad-accum", help="Gradient accumulation steps"),
    learning_rate: float = typer.Option(2e-4, "--learning-rate", "--lr", help="Learning rate"),
    lora_r: int = typer.Option(16, "--lora-r", help="LoRA rank dimension"),
    use_4bit: bool = typer.Option(False, "--4bit", help="Enable 4-bit QLoRA (requires CUDA)"),
):
    """Fine-tune Qwen model with PEFT LoRA on Indian SMS extraction."""
    from slm_sms.data.dataset import load_hf_dataset_from_dir
    from slm_sms.training.trainer import SMSTrainingPipeline

    console.print(f"[bold blue]Starting Fine-Tuning Pipeline for {model_id}[/bold blue]")
    dataset = load_hf_dataset_from_dir(data_dir)

    pipeline = SMSTrainingPipeline(
        model_id=model_id,
        output_dir=output_dir,
        lora_r=lora_r,
        use_4bit=use_4bit,
    )

    metrics = pipeline.train(
        dataset=dataset,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
    )
    console.print(f"[bold green]Training complete! Metrics: {metrics}[/bold green]")


@app.command("infer")
def infer_cli(
    text: str = typer.Argument(..., help="Indian SMS text to analyze"),
    model_id: str = typer.Option("Qwen/Qwen3.5-2B", "--model-id", "-m", help="Base model ID"),
    adapter_path: str | None = typer.Option(
        None, "--adapter-path", "-a", help="Path to LoRA adapter"
    ),
):
    """Run entity extraction on a single SMS text."""
    from slm_sms.inference.engine import SMSExtractorEngine

    console.print(f"[bold cyan]Analyzing SMS:[/bold cyan] {text}")
    engine = SMSExtractorEngine(model_id_or_path=model_id, adapter_path=adapter_path)
    result = engine.extract(text)

    console.print("\n[bold green]Extracted Entities (JSON):[/bold green]")
    console.print_json(json.dumps(result["entity"], indent=2))
    console.print(
        f"[dim]Latency: {result['latency_ms']} ms | Fallback used: {result['fallback_used']}[/dim]"
    )


@app.command("evaluate")
def evaluate_cli(
    test_file: Path = typer.Option(
        Path("./data/test.jsonl"), "--test-file", "-t", help="Test JSONL file"
    ),
    model_id: str = typer.Option("Qwen/Qwen3.5-2B", "--model-id", "-m", help="Base model ID"),
    adapter_path: str | None = typer.Option(
        None, "--adapter-path", "-a", help="Path to LoRA adapter"
    ),
    max_samples: int = typer.Option(50, "--max-samples", help="Max evaluation samples"),
):
    """Benchmark extraction accuracy and JSON validity on test dataset."""
    from slm_sms.data.dataset import load_jsonl
    from slm_sms.inference.engine import SMSExtractorEngine
    from slm_sms.training.metrics import compute_extraction_metrics

    records = load_jsonl(test_file)[:max_samples]
    engine = SMSExtractorEngine(model_id_or_path=model_id, adapter_path=adapter_path)

    predictions = []
    ground_truths = []

    with console.status(f"[bold blue]Evaluating {len(records)} test samples..."):
        for rec in records:
            sms_text = rec["messages"][1]["content"]
            gt_json = rec["messages"][2]["content"]
            result = engine.extract(sms_text)
            predictions.append(result["raw_output"])
            ground_truths.append(json.loads(gt_json))

    report = compute_extraction_metrics(predictions, ground_truths)
    table = Table(title="Extraction Benchmark Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for k, v in report.to_dict().items():
        table.add_row(k, str(v))

    console.print(table)


@app.command("serve")
def serve_cli(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
    model_id: str = typer.Option("Qwen/Qwen3.5-2B", "--model-id", help="Model ID"),
    adapter_path: str | None = typer.Option(None, "--adapter-path", help="Adapter path"),
):
    """Launch production FastAPI HTTP microservice."""
    import os

    import uvicorn

    os.environ["SLM_MODEL_ID"] = model_id
    if adapter_path:
        os.environ["SLM_ADAPTER_PATH"] = adapter_path

    console.print(f"[bold green]Starting SLM SMS Server on http://{host}:{port}[/bold green]")
    uvicorn.run("slm_sms.inference.server:app", host=host, port=port, workers=workers)


def run_generate():
    typer.run(generate_cli)


def run_train():
    typer.run(train_cli)


def run_infer():
    typer.run(infer_cli)


def run_evaluate():
    typer.run(evaluate_cli)


def run_serve():
    typer.run(serve_cli)


if __name__ == "__main__":
    app()
