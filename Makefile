.PHONY: setup install test test-live lint collect collect-leagues train evaluate tune scrape picks clean help

PYTHON = .venv/bin/python
CLI = $(PYTHON) -m src.app.cli

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Full setup: venv + deps + playwright browser
	uv venv
	uv pip install -e ".[dev]"
	$(PYTHON) -m playwright install chromium
	@echo "\n✓ Setup complete. Run 'source .venv/bin/activate' or use 'make' commands."

install: ## Install deps only (assumes venv exists)
	uv pip install -e ".[dev]"

test: ## Run test suite
	$(PYTHON) -m pytest tests/

test-live: ## Run live scraper tests (hits real sites, verifies DOM selectors)
	$(PYTHON) -m pytest tests/test_scrapers.py -m live -v -o "addopts="

test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing

lint: ## Lint code with ruff
	$(PYTHON) -m ruff check src/ tests/

lint-fix: ## Auto-fix lint issues
	$(PYTHON) -m ruff check --fix src/ tests/

collect: ## Fetch NBA game data (3 seasons)
	$(CLI) -v collect

collect-leagues: ## Scrape international league results from Flashscore
	$(CLI) -v collect-leagues --no-headless

train: ## Train the margin prediction model (all leagues)
	$(CLI) -v train

train-ridge: ## Train with Ridge backend
	$(CLI) -v train --model ridge

train-rf: ## Train with Random Forest backend
	$(CLI) -v train --model rf

evaluate: ## Run walk-forward backtesting
	$(CLI) -v evaluate

evaluate-all: ## Evaluate all backends side by side
	@echo "=== XGBoost ===" && $(CLI) evaluate --model xgboost
	@echo "\n=== Ridge ===" && $(CLI) evaluate --model ridge
	@echo "\n=== Random Forest ===" && $(CLI) evaluate --model rf

tune: ## Tune hyperparameters (default: xgboost, 30 trials)
	$(CLI) -v tune

scrape: ## Scrape winning margin odds from SportsPlus
	$(CLI) -v scrape --no-headless

picks: ## Show today's +EV picks
	$(CLI) picks

pipeline: collect train scrape picks ## Run full pipeline: collect → train → scrape → picks

pipeline-full: collect collect-leagues train scrape picks ## Full pipeline with international leagues

clean: ## Remove generated data and caches
	rm -rf .pytest_cache htmlcov .coverage
	rm -rf data/sportsbetting.db data/models/*.pkl
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
