# Contributing

Thanks for your interest in contributing! This project is in active early development and contributions are welcome.

## Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/sportsbetting.git
cd sportsbetting
make setup          # creates venv, installs deps, downloads Chromium
source .venv/bin/activate
```

> **macOS users:** XGBoost requires OpenMP. If you get a `libxgboost.dylib` error, run `brew install libomp`.

## Development Workflow

```bash
make test           # run the test suite (160+ tests)
make lint           # check code style with ruff
make lint-fix       # auto-fix lint issues
make test-cov       # tests with coverage report
```

### Before Submitting a PR

1. **Write tests** for any new functionality — see `tests/` for patterns
2. **Run the full suite:** `make test`
3. **Lint your code:** `make lint`
4. Keep commits focused on a single change

## Project Layout

```
src/
├── data/           # Data collection and storage
├── features/       # Feature engineering (Elo, rolling stats, context)
├── model/          # XGBoost model + folded normal distribution
├── betting/        # Value calculation, Kelly sizing, bet tracking
└── app/            # CLI and (future) dashboard
```

Each module is fairly independent. The data flows like this:

```
nba_api / scrapers → DataStore (SQLite) → feature builder → model → Platt calibration → value calculator → CLI output
```

Configuration is centralized in `config/settings.yaml` and loaded via `src/config.py`. Modules use `cfg("section", "key", default)` to read settings, with hardcoded fallbacks if the YAML is missing.

## Where to Contribute

### Good First Issues

- **Add more features:** Win streak, head-to-head history, pace-of-play
- **Better rest day calculation:** Use actual schedule data instead of date diff
- **Improve CLI output:** Add color, formatting, or export to CSV/JSON

### Medium Effort

- **Backtest reporting:** Generate plots and detailed ROI analysis per bucket
- **Edge threshold tuning:** Compare 3% / 5% / 7% / 10% thresholds vs ROI tradeoff

### Larger Projects

- **Streamlit dashboard** (Phase 4): Daily picks display with bet tracking
- **API-Basketball integration:** Replace web scraping with API for international leagues
- **Live odds monitoring:** Detect line movements and re-calculate value in real-time

## Code Style

- Python 3.11+, type hints encouraged
- Ruff for linting (config in `pyproject.toml`)
- Keep functions focused and under ~50 lines where possible
- Docstrings for public functions (Google style)
- Tests go in `tests/test_<module>.py` mirroring the `src/` structure

## Architecture Decisions

- **SQLite over Postgres:** Keeps the project portable and zero-config. The data volume (~10K games) doesn't need a server DB.
- **XGBoost over neural nets:** Simpler, faster, more interpretable for tabular features. Easy to inspect feature importance.
- **Folded normal distribution:** Converts a single point prediction into a full probability distribution over margin buckets. This is the key insight — see `src/model/distribution.py`.
- **Platt scaling:** Calibrates raw bucket probabilities via logistic regression to correct systematic over/under-confidence. Fitted during `train` and applied automatically during prediction.
- **Centralized config:** All tunable parameters in `config/settings.yaml`, loaded once via `src/config.py`. Modules accept `None` defaults and resolve from config at runtime, so explicit arguments still override.
- **1/4 Kelly sizing:** Conservative stake sizing that sacrifices ~25% of theoretical growth for ~75% variance reduction. Important when model probabilities have estimation error.

## Questions?

Open an issue or start a discussion. We're happy to help you get oriented in the codebase.
