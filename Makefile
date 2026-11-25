.PHONY: install test test-unit test-integration lint format serve serve-prod load-test clean

PYTHON ?= .venv/bin/python

install:
	uv sync --extra notebook --extra dev

test:
	$(PYTHON) -m pytest -v

test-unit:
	$(PYTHON) -m pytest tests/unit -v

test-integration:
	$(PYTHON) -m pytest tests/integration -v

lint:
	$(PYTHON) -m ruff check src/ scripts/ tests/
	$(PYTHON) -m ruff format --check src/ scripts/ tests/

format:
	$(PYTHON) -m ruff check --fix src/ scripts/ tests/
	$(PYTHON) -m ruff format src/ scripts/ tests/

serve:
	$(PYTHON) scripts/serve.py --reload

serve-prod:
	uv run gunicorn slm_sms.inference.server:app -c gunicorn.conf.py

load-test:
	uv run locust -f tests/load/locustfile.py --host http://localhost:8000

clean:
	rm -rf .coverage htmlcov .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
