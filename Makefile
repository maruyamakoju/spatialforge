# ============================================================
# SpatialForge — Makefile
# ============================================================

.PHONY: help dev test lint format docker-up docker-down setup-models benchmark clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Development ──────────────────────────────────────────────

dev: ## Start API server in development mode (auto-reload)
	DEBUG=true uvicorn spatialforge.main:app --host 0.0.0.0 --port 8000 --reload

worker: ## Start Celery worker
	celery -A spatialforge.workers.celery_app worker --loglevel=info --concurrency=1 -Q default,gpu_heavy

worker-beat: ## Start Celery beat scheduler (for periodic tasks)
	celery -A spatialforge.workers.celery_app beat --loglevel=info

# ── Testing ──────────────────────────────────────────────────

test: ## Run all tests
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=spatialforge --cov-report=html --cov-report=term

# ── Code Quality ─────────────────────────────────────────────

lint: ## Run linter (ruff)
	ruff check spatialforge/ tests/

format: ## Auto-format code
	ruff check --fix spatialforge/ tests/
	ruff format spatialforge/ tests/

typecheck: ## Run type checker
	mypy spatialforge/

# ── Docker ───────────────────────────────────────────────────

docker-up: ## Start all services via Docker Compose
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-build: ## Build Docker images
	docker compose build

docker-logs: ## Follow all service logs
	docker compose logs -f

# ── Models ───────────────────────────────────────────────────

setup-models: ## Download default depth model (giant)
	python scripts/setup_models.py --model giant

setup-models-all: ## Download all depth models
	python scripts/setup_models.py --model all

# ── Benchmarks ───────────────────────────────────────────────

benchmark: ## Run inference benchmark (default: giant @ 1080p)
	python scripts/benchmark.py --model giant --resolution 1080

benchmark-all: ## Benchmark all model sizes
	python scripts/benchmark.py --model all

# ── Admin ────────────────────────────────────────────────────

create-key: ## Create a new API key (usage: make create-key OWNER=myapp PLAN=builder)
	@curl -s -X POST "http://localhost:8000/v1/admin/keys?owner=$(OWNER)&plan=$(PLAN)" \
		-H "X-API-Key: $${ADMIN_API_KEY:-sf_admin_change_me}" | python -m json.tool

# ── Cleanup ──────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ htmlcov/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── Install ──────────────────────────────────────────────────

install: ## Install package in development mode
	pip install -e ".[dev]"

install-sdk: ## Install the Python SDK
	pip install -e sdk/
