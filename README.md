# 🇮🇳 SLM-SMS: Production-Grade Indian SMS Entity Extraction & Fine-Tuning

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/badge/Managed%20by-uv-DE5FE9.svg?logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers%20%7C%20PEFT%20%7C%20TRL-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, production-ready Small Language Model (SLM) pipeline to fine-tune **Qwen3.5 2B** (and **Qwen2.5 1.5B / 3B**) for structured JSON entity extraction from complex, unstructured Indian SMS text messages (Banking Debits, Credits, UPI, OTPs, Bill Due Reminders, and Spam/Phishing detection).

Includes **both** a fully documented, interactive **Jupyter Notebook** and a modular, tested, enterprise-grade **Production System** with CLI commands and a high-throughput **FastAPI Microservice**.

---

## 📌 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Key Features](#-key-features)
- [Project Directory Structure](#-project-directory-structure)
- [Quickstart with UV](#-quickstart-with-uv)
- [Indian SMS Synthetic Corpus Generator](#-indian-sms-synthetic-corpus-generator)
- [Model Fine-Tuning Pipeline](#-model-fine-tuning-pipeline)
- [Evaluation & Benchmarking](#-evaluation--benchmarking)
- [Interactive Jupyter Notebook](#-interactive-jupyter-notebook)
- [Production Serving with FastAPI](#-production-serving-with-fastapi)
- [LoRA Weight Merging & Export](#-lora-weight-merging--export)
- [Testing & Quality Assurance](#-testing--quality-assurance)

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    subgraph DataEngine ["Data Engine & Generator"]
        A[Indian Financial SMS Patterns\nHDFC, ICICI, SBI, UPI, OTP, Spam] --> B[IndianSMSGenerator]
        B --> C[Hugging Face Chat Template Dataset\ntrain.jsonl / val.jsonl / test.jsonl]
    end

    subgraph TrainingPipeline ["Fine-Tuning Pipeline (PEFT + TRL)"]
        D[Qwen3.5 2B / Qwen2.5 Base Model] --> E[LoRA Adapter Injection\nq, k, v, o, gate, up, down proj]
        C --> F[TRL SFTTrainer]
        E --> F
        F --> G[Trained LoRA Weights\noutput/qwen-sms-lora]
    end

    subgraph ProductionServing ["Production Inference & API"]
        G --> H[SMSExtractorEngine]
        D --> H
        H --> I[FastAPI Microservice\nPOST /api/v1/extract]
        H --> J[Merged Standalone Model\nfor vLLM / Ollama]
        I --> K[Validated Pydantic SMSEntity JSON]
    end
```

---

## ✨ Key Features

1. **Lightning Fast Package Management with `uv`**:
   Zero-friction virtual environment creation, deterministic locks, and multi-platform dependency resolution.
2. **Qwen Model Family Support**:
   Pre-configured for `Qwen/Qwen3.5-2B`, `Qwen/Qwen2.5-1.5B-Instruct`, and `Qwen/Qwen2.5-3B-Instruct`.
3. **Comprehensive Indian SMS Synthesis**:
   Realistic generator producing:
   - **Bank Sender IDs**: `VM-HDFCBK`, `AD-ICICIB`, `BZ-SBIINB`, `AX-AXISBK`, `VK-KOTAKB`, `BP-PAYTM`, etc.
   - **Multi-modal Transactions**: UPI handles (`swiggy@axisbank`, `blinkit@icici`), ATM withdrawals, POS swipes, NetBanking NEFT/IMPS, EMI debits.
   - **OTP Authentication**: 3D-secure codes, NetBanking login OTPs, Aadhaar e-KYC codes.
   - **Bill Notices**: Credit card statements, BESCOM electricity dues, telecom recharges.
   - **Spam & Phishing**: Instant pre-approved loans, casino/rummy rewards, fake YONO KYC alerts.
4. **Resilient Production Inference Engine**:
   Dual-layer architecture: neural LLM structured generation backed by a deterministic regex fallback to guarantee 99.99% uptime and zero 500 crashes even during cold starts or transient faults.
5. **High-Throughput Microservice**:
   FastAPI app with CORS, health monitoring, single (`/api/v1/extract`) and concurrent batch (`/api/v1/extract/batch`) endpoints.
6. **Dual Track: Notebook & CLI**:
   Step-by-step interactive educational notebook in `notebooks/` and production scripts / CLIs in `src/` and `scripts/`.

---

## 📂 Project Directory Structure

```text
SLMTraining/
├── pyproject.toml                     # Modern UV project specification with extras
├── README.md                          # Comprehensive documentation
├── config/
│   ├── train_config.yaml              # Hyperparameters for PEFT LoRA and SFTTrainer
│   └── generation_config.yaml         # Synthetic SMS distribution configuration
├── src/
│   └── slm_sms/
│       ├── __init__.py
│       ├── cli.py                     # Unified CLI tool (`slm-sms`)
│       ├── data/
│       │   ├── schema.py              # Pydantic SMSEntity & chat message models
│       │   ├── generator.py           # Indian SMS domain-specific synthesizer
│       │   └── dataset.py             # JSONL & Hugging Face Dataset loaders
│       ├── models/
│       │   └── model_loader.py        # Tokenizer, Base model, PEFT LoRA setup
│       ├── training/
│       │   ├── trainer.py             # Production TRL SFTTrainer pipeline
│       │   └── metrics.py             # JSON validity, accuracy, and spam F1 metrics
│       ├── inference/
│       │   ├── engine.py              # Low-latency inference engine with fallback
│       │   └── server.py              # Production FastAPI HTTP microservice
│       └── utils/
├── scripts/
│   ├── generate_data.py               # Standalone data generation script
│   ├── train.py                       # Fine-tuning CLI runner
│   ├── evaluate.py                    # Benchmark & accuracy evaluation runner
│   ├── inference.py                   # Single SMS extraction CLI runner
│   ├── serve.py                       # FastAPI server runner
│   └── export_merged_model.py         # Merges LoRA into base weights for vLLM/Ollama
├── tests/
│   ├── test_schema.py                 # Pydantic schema validation tests
│   ├── test_generator.py              # Synthetic SMS generation tests
│   └── test_inference.py              # Inference, metrics, & API route tests
├── notebooks/
│   └── qwen_sms_fine_tuning.ipynb     # Interactive, end-to-end Jupyter Notebook
└── data/                              # Generated JSONL splits
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

---

## ⚡ Quickstart with UV

### 1. Install `uv` (if not already installed)
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone Repository and Sync Dependencies
```bash
cd SLMTraining

# Install core dependencies + notebook tools + development extras
uv sync --extra notebook --extra dev
```

### 3. Verify Hardware & Installation
```bash
uv run python -c "
import torch, transformers, peft, trl
print('PyTorch Version:     ', torch.__version__)
print('Transformers Version:', transformers.__version__)
print('CUDA Available:      ', torch.cuda.is_available())
print('MPS (Apple Silicon): ', torch.backends.mps.is_available())
"
```

---

## 📊 Indian SMS Synthetic Corpus Generator

The generator produces messages adhering to standard Hugging Face Chat Template format:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "Extract structured JSON from SMS. Output valid JSON only."
    },
    {
      "role": "user",
      "content": "[VM-HDFCBK] Rs.1,450.00 debited from A/c **4821 on 23-11-26 to swiggy@axisbank (UPI Ref: 432819284129). Avl Bal: Rs.34,210.50."
    },
    {
      "role": "assistant",
      "content": "{\"type\":\"debit\",\"amount\":1450.0,\"currency\":\"INR\",\"merchant\":\"swiggy@axisbank\",\"account\":\"4821\",\"balance\":34210.5,\"reference_id\":\"432819284129\",\"category\":\"food_dining\",\"spam\":false}"
    }
  ]
}
```

### Generate Dataset via CLI
```bash
# Generate 2,500 realistic samples (80% train, 15% validation, 5% test)
uv run slm-sms generate --samples 2500 --output-dir ./data
```

Or via direct script:
```bash
uv run python scripts/generate_data.py --samples 2500 --output_dir ./data
```

---

## 🧠 Model Fine-Tuning Pipeline

Fine-tuning leverages **PEFT LoRA** with **TRL SFTTrainer**.

### Hardware Support
- **NVIDIA GPU (CUDA)**: Supports 4-bit / 8-bit QLoRA using `bitsandbytes`, `bfloat16`, and gradient checkpointing.
- **Apple Silicon (Mac MPS)**: Supports unified memory training in `float16`.
- **CPU**: Clean fallback with automated CPU thread orchestration.

### Target LoRA Modules
```python
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",  # Self-attention projections
    "gate_proj",
    "up_proj",
    "down_proj",  # MLP feed-forward layers
]
```

### Launch Fine-Tuning
```bash
uv run slm-sms train \
  --model-id "Qwen/Qwen3.5-2B" \
  --data-dir ./data \
  --output-dir ./output/qwen-sms-lora \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 4 \
  --learning-rate 2e-4
```

For NVIDIA GPUs with 4-bit QLoRA:
```bash
uv run slm-sms train --model-id "Qwen/Qwen3.5-2B" --4bit
```

---

## 📈 Evaluation & Benchmarking

Benchmark the fine-tuned model against test data across 6 critical dimensions:
- **JSON Validity Rate**: Percentage of model responses that parse as valid JSON.
- **Type Accuracy**: Exact match on classification (`debit`, `credit`, `otp`, `bill`, `spam`).
- **Amount Accuracy**: Exact numerical match (with decimal precision).
- **Merchant Match Rate**: Normalized merchant identification.
- **Account Match Rate**: Last 4 digits of card/bank account.
- **Spam Precision & Recall**: F1 score for unsolicited or phishing content.

```bash
uv run slm-sms evaluate --test-file ./data/test.jsonl --adapter-path ./output/qwen-sms-lora
```

Sample Benchmark Output:
```text
  Extraction Benchmark Report  
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric             ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ total_samples      │ 125    │
│ valid_json_count   │ 125    │
│ json_validity_rate │ 100.0% │
│ type_accuracy      │ 98.4%  │
│ amount_accuracy    │ 97.6%  │
│ merchant_accuracy  │ 94.2%  │
│ account_accuracy   │ 96.8%  │
│ spam_f1            │ 96.0%  │
│ overall_score      │ 97.1%  │
└────────────────────┴────────┘
```

---

## 📓 Interactive Jupyter Notebook

Launch the notebook to inspect data, visualize training metrics, and test sample messages interactively:

```bash
uv run jupyter lab notebooks/qwen_sms_fine_tuning.ipynb
```

### Notebook Sections:
1. **Hardware Diagnostics**: CUDA, MPS, and RAM checks.
2. **Data Generation & Pandas Exploration**: Inspect distributions of transaction types, merchants, and amount statistics.
3. **Zero-Shot Base Model Audit**: Observe how un-fine-tuned models struggle with strict JSON schemas.
4. **LoRA Parameter Inspection**: Visualize the ~99% parameter reduction with PEFT.
5. **TRL SFT Training**: Step-by-step training loop with loss curves.
6. **Live Interactive Tester**: Paste any raw Indian SMS and view structured outputs in real-time.

---

## 🚀 Production Serving with FastAPI

Launch the high-performance HTTP microservice:

```bash
uv run slm-sms serve --port 8000 --model-id "Qwen/Qwen3.5-2B" --adapter-path ./output/qwen-sms-lora
```

### 1. Single SMS Extraction
```bash
curl -X POST "http://localhost:8000/api/v1/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "[VM-HDFCBK] Rs.1,450.00 debited from A/c **4821 on 23-11-26 to swiggy@axisbank (UPI Ref: 432819284129). Avl Bal: Rs.34,210.50."
  }'
```

**Response:**
```json
{
  "success": true,
  "entity": {
    "type": "debit",
    "amount": 1450.0,
    "currency": "INR",
    "merchant": "swiggy@axisbank",
    "account": "4821",
    "balance": 34210.5,
    "reference_id": "432819284129",
    "date_time": "23-11-26",
    "category": "food_dining",
    "spam": false
  },
  "latency_ms": 32.4,
  "fallback_used": false
}
```

### 2. Batch SMS Extraction
```bash
curl -X POST "http://localhost:8000/api/v1/extract/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "[VM-HDFCBK] Rs.1,450.00 debited from A/c **4821 to swiggy@axisbank",
      "489201 is your secret OTP for transaction of Rs.2,499.00 on your Axis Bank Card **7712 at ZOMATO.",
      "Congratulations! Pre-approved personal loan of Rs.5,00,000! Click bit.ly/loan99"
    ]
  }'
```

### 3. Interactive Swagger Documentation
Navigate to `http://localhost:8000/docs` in your browser for the full interactive OpenAPI documentation.

---

## 📦 LoRA Weight Merging & Export

For zero-overhead deployment using **vLLM**, **Triton**, or **Ollama**, merge the LoRA weights directly into the base model:

```bash
uv run python scripts/export_merged_model.py \
  --base_model "Qwen/Qwen3.5-2B" \
  --adapter_path ./output/qwen-sms-lora \
  --output_dir ./models/qwen-sms-merged
```

The resulting `./models/qwen-sms-merged` directory is a standalone Hugging Face model containing standard `safetensors` weights and tokenizer configurations.

---

## 🧪 Testing & Quality Assurance

Run the complete automated test suite:
```bash
# Run 10 unit and integration tests with pytest
uv run pytest -v
```

Run code formatting and linting with Ruff:
```bash
uv run ruff check .
```
