.DEFAULT_GOAL := help

PYTHON           ?= .venv/bin/python
UV               ?= uv

MODEL_ID         ?= Qwen/Qwen3.5-2B
DATA_DIR         ?= ./data
OUTPUT_DIR       ?= ./output/qwen-sms-lora
MERGED_DIR       ?= ./models/qwen-sms-merged
ADAPTER_PATH     ?=
GEN_CONFIG       ?= config/generation_config.yaml
TRAIN_CONFIG     ?= config/train_config.yaml

SAMPLES          ?= 2500
MAX_EVAL_SAMPLES ?= 50
PORT             ?= 8000
OTEL_ENDPOINT    ?= http://localhost:4318
SMS              ?= [VM-HDFCBK] Rs.1,450.00 debited from A/c **4821 on 23-11-26 to swiggy@axisbank (UPI Ref: 432819284129). Avl Bal: Rs.34,210.50.

.PHONY: help install install-core install-cuda lock check-hw \
	generate-data generate-data-config train-model train-qlora \
	evaluate infer export-model notebook \
	serve serve-otel serve-console-otel serve-prod load-test load-test-headless \
	test test-unit test-integration coverage lint format clean clean-outputs

help: ## Show available Make commands
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Setup & Environment
# ---------------------------------------------------------------------------

install: ## Install core, notebook, and dev dependencies via uv
	$(UV) sync --extra notebook --extra dev

install-core: ## Install core runtime dependencies only
	$(UV) sync

install-cuda: ## Install dependencies including CUDA bitsandbytes (Linux GPU)
	$(UV) sync --extra cuda --extra notebook --extra dev

lock: ## Refresh uv.lock from pyproject.toml
	$(UV) lock

check-hw: ## Verify Python, PyTorch, Transformers, CUDA, and MPS availability
	$(PYTHON) -c "import sys, torch, transformers, peft, trl; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__, '| Transformers:', transformers.__version__); print('CUDA:', torch.cuda.is_available(), '| MPS:', torch.backends.mps.is_available())"

# ---------------------------------------------------------------------------
# Data, Fine-Tuning & Evaluation
# ---------------------------------------------------------------------------

generate-data: ## Generate synthetic SMS splits (override SAMPLES=5000 DATA_DIR=./data)
	$(PYTHON) scripts/generate_data.py --samples $(SAMPLES) --output_dir $(DATA_DIR)

generate-data-config: ## Generate synthetic SMS splits from GEN_CONFIG YAML
	$(PYTHON) scripts/generate_data.py --config $(GEN_CONFIG)

train-model: ## Fine-tune Qwen with LoRA using config/train_config.yaml defaults
	$(PYTHON) scripts/train.py --config $(TRAIN_CONFIG) --data_dir $(DATA_DIR)

train-qlora: ## Fine-tune Qwen with 4-bit QLoRA (NVIDIA CUDA GPU required)
	$(PYTHON) scripts/train.py --config $(TRAIN_CONFIG) --data_dir $(DATA_DIR) --use_4bit

evaluate: ## Evaluate extraction accuracy on test split (optional ADAPTER_PATH=...)
	$(PYTHON) scripts/evaluate.py --test_file $(DATA_DIR)/test.jsonl --model_id $(MODEL_ID) --max_samples $(MAX_EVAL_SAMPLES) $(if $(ADAPTER_PATH),--adapter_path $(ADAPTER_PATH),)

infer: ## Extract entities from single SMS (override SMS="..." ADAPTER_PATH=...)
	$(PYTHON) scripts/inference.py --text "$(SMS)" --model_id $(MODEL_ID) $(if $(ADAPTER_PATH),--adapter_path $(ADAPTER_PATH),)

export-model: ## Merge trained LoRA adapter into standalone Hugging Face weights
	$(PYTHON) scripts/export_merged_model.py --base_model $(MODEL_ID) --adapter_path $(OUTPUT_DIR) --output_dir $(MERGED_DIR)

notebook: ## Launch interactive JupyterLab training & exploration notebook
	$(UV) run jupyter lab notebooks/qwen_sms_fine_tuning.ipynb

# ---------------------------------------------------------------------------
# Serving & Load Testing
# ---------------------------------------------------------------------------

serve: ## Start FastAPI dev server with auto-reload (uses .env)
	$(PYTHON) scripts/serve.py --port $(PORT) --reload

serve-otel: ## Start FastAPI server exporting OTel traces & metrics to OTEL_ENDPOINT
	SLM_OTEL_EXPORTER=otlp OTEL_EXPORTER_OTLP_ENDPOINT=$(OTEL_ENDPOINT) $(PYTHON) scripts/serve.py --port $(PORT)

serve-console-otel: ## Start FastAPI server printing OTel traces & metrics to stdout
	SLM_OTEL_EXPORTER=console $(PYTHON) scripts/serve.py --port $(PORT)

serve-prod: ## Start production Gunicorn + UvicornWorker server (gunicorn.conf.py)
	$(UV) run gunicorn slm_sms.inference.server:app -c gunicorn.conf.py

load-test: ## Open Locust web UI at http://localhost:8089
	$(UV) run locust -f tests/load/locustfile.py --host http://localhost:$(PORT)

load-test-headless: ## Run 60s headless Locust load benchmark (50 users)
	$(UV) run locust -f tests/load/locustfile.py --host http://localhost:$(PORT) --users 50 --spawn-rate 10 --run-time 60s --headless

# ---------------------------------------------------------------------------
# Testing, Quality & Cleanup
# ---------------------------------------------------------------------------

test: ## Run all unit and integration tests with coverage
	$(PYTHON) -m pytest -v

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit -v

test-integration: ## Run FastAPI HTTP integration tests only
	$(PYTHON) -m pytest tests/integration -v

coverage: ## Generate terminal & HTML coverage report (htmlcov/index.html)
	$(PYTHON) -m pytest --cov-report=term-missing --cov-report=html:htmlcov -q

lint: ## Run Ruff lint and formatting checks
	$(PYTHON) -m ruff check src/ scripts/ tests/
	$(PYTHON) -m ruff format --check src/ scripts/ tests/

format: ## Auto-fix Ruff lint issues and format code
	$(PYTHON) -m ruff check --fix src/ scripts/ tests/
	$(PYTHON) -m ruff format src/ scripts/ tests/

clean: ## Remove test, coverage, and Python cache files
	rm -rf .coverage htmlcov .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-outputs: ## Remove generated model training and export artifacts (output/, models/)
	rm -rf output models
