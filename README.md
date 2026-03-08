# Basketball Winning Margin Predictor

A sports betting model that predicts **winning margin bucket probabilities** for basketball games and identifies positive expected value (+EV) bets by comparing model predictions against bookmaker odds.

**Target market:** "Any Team Winning Margin" — buckets like 1-5, 6-10, 11-15, 16-20, 21-25, 26-30, 31+ (odds typically range from ~3x to ~25x).

```
=====================================================================================
  PICKS — 2026-03-06  |  Bankroll: $10,000  |  Min Edge: 3%
=====================================================================================
Game                                Bucket  Model%   Book%   Edge EV/unit    Stake
-------------------------------------------------------------------------------------
Phoenix Suns vs Chicago Bulls          1-5  41.9%  25.2% +16.7%  +0.66 $    300
Houston Rockets vs Golden State...     1-5  34.3%  28.8% +5.5%  +0.19 $    193
Denver Nuggets vs Los Angeles L...   21-25  16.3%   9.8% +6.5%  +0.67 $    181
Miami Heat vs Brooklyn Nets          21-25  20.0%  14.5% +5.5%  +0.38 $    162
Denver Nuggets vs Los Angeles L...   16-20  21.0%  15.8% +5.2%  +0.33 $    153
-------------------------------------------------------------------------------------
Total exposure:                                                 $  1,500 (15.0%)
```

## How It Works

```
Historical Game Data ─┐
  NBA (nba_api)       ├→ Feature Engineering → Preprocessor → ML Backend → Folded Normal → Platt Calibration → Bucket Probabilities
  30 Leagues          │   (Elo, form, rest,    (scale/one-hot   (XGBoost,    (per-league σ)   (per-bucket LR)            ↓
  (Flashscore)  ──────┘    schedule, injuries)  per backend)    Ridge, RF)                                               ↓
                                                                                                                         ↓
SportsPlus Odds → Implied Probabilities ──────────────────────────────────→ Edge Calculation ──────────→ +EV Picks
```

1. **Collect game data** from multiple sources — NBA via `nba_api`, 30+ international leagues via Flashscore scraping. All stored in a single SQLite database.
2. **Predict the signed margin** (e.g., "home team wins by 5") using a pluggable ML backend trained per league — `auto-tune` evaluates all backends (XGBoost, Ridge, RF) per league via walk-forward CV, picks the best, and tunes its hyperparameters with Optuna
3. **Convert to a probability distribution** over absolute margin using a [folded normal distribution](https://en.wikipedia.org/wiki/Folded_normal_distribution) with per-league σ — the key insight is that |N(μ, σ)| naturally models "any team wins by X"
4. **Calibrate probabilities** via [Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling) — fits a logistic regression per bucket to correct systematic over/under-confidence
5. **Scrape bookmaker odds** from SportsPlus.ph for the "Any Team Winning Margin" market
6. **Find value** where model probability exceeds implied probability by a configurable threshold
7. **Size bets** using 1/4 Kelly criterion with exposure caps

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- macOS: `brew install libomp` (required by XGBoost)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/sportsbetting.git
cd sportsbetting
make setup    # creates venv, installs deps, downloads Chromium for scraping
```

<details>
<summary>Manual setup (without make/uv)</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m playwright install chromium
```
</details>

### Run the Pipeline

```bash
source .venv/bin/activate

# Step 1: Collect historical NBA data (~3,700 games, takes ~5 seconds)
make collect

# Step 1b: Collect international league data from Flashscore (current + 2 past seasons)
python -m src.app.cli collect-leagues --seasons 2 --parallel 3

# Step 2: Auto-tune + train (picks best backend per league, tunes hyperparams)
make auto-tune

# Or train with a single backend:
# make train

# Step 3: Scrape today's odds from SportsPlus (~3 min with parallel scraping)
make scrape

# Step 4: See picks
make picks
```

Or run everything at once:

```bash
make pipeline           # NBA only: collect → train → scrape → picks
make pipeline-full      # All leagues: collect → collect-leagues → train → scrape → picks
```

### CLI Reference

```bash
python -m src.app.cli --help

Commands:
  collect          Fetch NBA game data and store it
  collect-leagues  Scrape international league results from Flashscore
  collect-players  Fetch NBA player game logs (for injury impact features)
  train            Train the margin prediction model
  evaluate         Run walk-forward backtesting
  tune             Tune hyperparameters with Bayesian optimization (Optuna)
  auto-tune        Auto-select best backend per league + tune hyperparams
  scrape           Scrape winning margin odds from SportsPlus
  picks            Show +EV picks with Kelly-sized stakes
  status           Show bet tracking P&L summary

Options for collect-leagues:
  -s, --seasons N       Scrape N past seasons in addition to current (default: 0)
  -p, --parallel N      Scrape N leagues concurrently (default: 1)
  --incremental/--full  Skip re-scraping games already in DB (default: full)
  -c, --clicks N        Max "Show more" clicks per league (default: 20)
  --headless/--no-headless  Run browser headless (default: headless)

Options for train/evaluate:
  -l, --league          Train/evaluate on single league (default: all)
  -m, --model BACKEND   ML backend: xgboost|ridge|rf|lightgbm (default: xgboost)
  -p, --params JSON     Backend hyperparams as JSON string

Options for tune:
  -m, --model BACKEND   ML backend to tune (default: xgboost)
  -n, --trials N        Number of Optuna trials (default: 30)
  --metric mae|rmse     Metric to minimize (default: mae)
  -l, --league          Tune on single league (default: all)

Options for auto-tune:
  -n, --trials N        Optuna trials per league (default: 20)
  --metric mae|rmse     Metric to minimize (default: mae)
  --no-tune             Only compare backends, skip Optuna tuning

Options for scrape:
  -p, --parallel N      Games to scrape concurrently (default: 3)
  --headless/--no-headless  Run browser headless (default: headless)

Options for picks:
  -b, --bankroll  Current bankroll (default: 10000)
  -e, --min-edge  Minimum edge threshold (default: 0.05 = 5%)
  -k, --kelly     Kelly fraction (default: 0.25 = quarter Kelly)
  -d, --date      Game date YYYY-MM-DD (default: latest with odds)
  --out           Comma-separated player names to mark as out
  --no-injuries   Skip fetching injury report
```

## Project Structure

```
sportsbetting/
├── src/
│   ├── data/
│   │   ├── data_store.py           # SQLite storage for games, odds, bets, player logs
│   │   ├── nba_collector.py        # NBA stats API data collection
│   │   ├── flashscore_scraper.py   # Playwright scraper: retries, concurrency, historical seasons
│   │   ├── sportsplus_scraper.py   # Playwright scraper for bookmaker odds (parallel + retry)
│   │   ├── injury_scraper.py       # Rotowire injury report scraper
│   │   ├── league_matcher.py       # Fuzzy league + team name matching (rapidfuzz)
│   │   └── team_names.py           # NBA team name mapping
│   ├── features/
│   │   ├── team_strength.py        # Elo rating system (per-league)
│   │   ├── form.py                 # Rolling margin & scoring averages
│   │   ├── context.py              # Rest days, back-to-backs, schedule position
│   │   ├── player_impact.py        # Player impact scores from game logs
│   │   └── builder.py              # Combines all features + league_id encoding
│   ├── model/
│   │   ├── margin_model.py         # Per-league models with pluggable backends + Platt calibration
│   │   ├── backends/               # ML backend registry (XGBoost, Ridge, RF, LightGBM)
│   │   │   ├── __init__.py         # Lazy-import registry
│   │   │   ├── base.py             # BaseBackend ABC + PreprocessingConfig
│   │   │   ├── xgboost_backend.py  # XGBoost (default)
│   │   │   ├── ridge_backend.py    # Ridge regression (scaling + one-hot)
│   │   │   ├── rf_backend.py       # Random Forest
│   │   │   └── lightgbm_backend.py # LightGBM (optional dependency)
│   │   ├── preprocessor.py         # Configurable scaling + one-hot encoding per backend
│   │   ├── tuner.py                # Optuna hyperparameter optimization
│   │   ├── auto_tuner.py           # Per-league backend selection + auto-tuning pipeline
│   │   ├── distribution.py         # Folded normal → bucket probabilities
│   │   └── calibration.py          # Platt scaling for probability calibration
│   ├── betting/
│   │   ├── value_calculator.py     # Edge = model_prob - implied_prob
│   │   ├── kelly.py                # 1/4 Kelly criterion stake sizing
│   │   └── tracker.py              # Bet recording and P&L tracking
│   └── app/
│       └── cli.py                  # Click CLI entry point
│   config.py                       # Centralized settings loader (config/settings.yaml)
├── config/
│   ├── settings.yaml               # Model + betting configuration (single source of truth)
│   └── league_mappings.yaml        # 30+ league → Flashscore URL mappings
├── tests/                          # 239 unit tests + 6 live integration tests
├── data/                           # SQLite DB + saved models (gitignored)
├── notebooks/                      # Exploration notebooks
├── pyproject.toml
├── Makefile
└── CONTRIBUTING.md
```

## Features Used by the Model

| Feature | Description |
|---------|-------------|
| `elo_diff` | Home Elo + home advantage (100) - Away Elo |
| `home_elo` / `away_elo` | Elo ratings (K=20, start 1500, per-league) |
| `home_avg_margin` / `away_avg_margin` | Rolling 10-game average absolute margin |
| `home_avg_scored` / `away_avg_scored` | Rolling 10-game scoring average |
| `home_rest_days` / `away_rest_days` | Days since last game |
| `home_game_num` / `away_game_num` | Season game count (schedule fatigue) |
| `is_b2b_home` / `is_b2b_away` | Back-to-back flag (rest ≤ 1 day) |
| `home_missing_impact` / `away_missing_impact` | Cumulative impact of injured/out players |
| `league_id` | Categorical league identifier (NBA=0, Euroleague=1, ...) |

## Bankroll Management

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Kelly fraction | 1/4 | Sacrifices ~25% growth for ~75% variance reduction |
| Min stake | 0.5% bankroll | Minimum to make a bet worthwhile |
| Max stake | 3% bankroll | Cap against model overconfidence |
| Max exposure | 15% bankroll | Limit total risk on any single day |
| Stop-loss | 25% drawdown | Pause and review if bankroll drops 25% from peak |

## Model Performance

Walk-forward cross-validation on 3,485 NBA games (2022-2025):

| Metric | Value |
|--------|-------|
| MAE | 11.87 points |
| RMSE | 15.09 points |
| σ (residual std) | 9.30 (training) / 15.09 (CV) |

The model is not trying to predict exact margins — it's trying to produce **well-calibrated bucket probabilities** that are more accurate than the bookmaker's implied odds. Even small calibration advantages compound over hundreds of bets.

## Data Sources

| Source | Coverage | Cost |
|--------|----------|------|
| [nba_api](https://github.com/swar/nba_api) | NBA (20+ years of game logs) | Free |
| [Flashscore](https://flashscore.com) (Playwright scraper) | 30 international leagues, ~22k games (3 seasons) | Free |
| SportsPlus.ph (Playwright scraper) | Winning margin odds for upcoming games | Free |
| [API-Basketball](https://rapidapi.com/api-sports/api/api-basketball) (future) | 115+ leagues via REST API | $7.99/mo |

## Configuration

All tunable parameters live in `config/settings.yaml`:

```yaml
model:
  elo_k_factor: 20          # Elo K-factor
  elo_start: 1500           # Starting Elo rating
  rolling_window: 10        # Rolling stats window size

betting:
  edge_threshold: 0.05      # 5% minimum edge
  kelly_fraction: 0.25      # 1/4 Kelly
  min_stake_pct: 0.005      # 0.5% of bankroll
  max_stake_pct: 0.03       # 3% of bankroll
  max_exposure_pct: 0.15    # 15% total exposure
  default_bankroll: 10000
```

CLI defaults and all modules read from this file via `src/config.py`. You can still override any value via CLI flags.

## Testing

```bash
make test           # 239 unit tests, ~15 seconds
make test-live      # 6 live integration tests (hits Flashscore, ~60s)
make test-cov       # with coverage report
make lint           # ruff linting
```

Tests cover: data store CRUD, Elo math properties, distribution sum-to-one invariants, Kelly boundary conditions, margin-to-bucket mapping, model train/predict/save/load across multiple backends, backend registry, preprocessor transforms + serialization, Optuna search space sampling, team name mapping, league + team fuzzy matching, Platt calibration, centralized config loading, player impact scoring, injury scraping, scraper date parsing, retry logic, season discovery, bucket regex, per-league backend selection, mixed-backend save/load roundtrip, auto-tune pipeline, rate-limit error handling, and **live DOM selector validation** against Flashscore (catches site structure changes).

## Roadmap

### ~~Phase 2: International Leagues~~ ✅
- [x] Flashscore scraper for 30+ leagues (Euroleague, ACB, BBL, LNB, CBA, KBL, etc.)
- [x] League + team name fuzzy matching between SportsPlus ↔ Flashscore via `rapidfuzz`
- [x] `league_id` as categorical feature with per-league σ estimation
- [x] Live integration tests to detect Flashscore DOM changes
- [x] Historical season scraping via archive page discovery (`--seasons N`)
- [x] Concurrent league scraping with configurable parallelism (`--parallel N`)
- [x] Incremental mode — only inserts new games (`--incremental`)
- [x] Retry with exponential backoff, smart "load more" with empty-cycle detection
- [x] `domcontentloaded` waits instead of `networkidle` (eliminates ad/tracker timeouts)
- [ ] SofaScore fallback when Flashscore scraping fails

### Phase 3: Backtesting & Calibration
- [x] Walk-forward backtesting framework with proper train/test split (`evaluate` command)
- [x] Platt scaling calibration for bucket probabilities (auto-fitted during training)
- [x] Centralized configuration via `config/settings.yaml`
- [x] League + team fuzzy matching tests
- [ ] Backtest reports: hit rate per bucket, ROI curve, max drawdown
- [ ] Tune edge threshold (3% / 5% / 7% / 10%) vs number of bets vs ROI

### ~~Phase 4: Player Impact & Injury Awareness~~ ✅
- [x] Player game log collection (`collect-players` command)
- [x] Player impact scoring (weighted minutes, points, plus/minus)
- [x] `home_missing_impact` / `away_missing_impact` features from injury data
- [x] Rotowire injury report scraper (auto-fetched during `picks`)
- [x] Manual `--out` flag for overriding injury data

### ~~Phase 5: Modular Model Backends + Auto-Tuning~~ ✅
- [x] Pluggable ML backends: XGBoost (default), Ridge, Random Forest, LightGBM
- [x] Per-backend preprocessing (tree models: passthrough; linear: scaling + one-hot)
- [x] Backend registry with lazy imports (`--model` flag on `train`/`evaluate`)
- [x] Hyperparameter tuning via Optuna Bayesian optimization (`tune` command)
- [x] `--params` JSON flag for training with custom/tuned hyperparameters
- [x] **Per-league auto-tune** (`auto-tune` command): compares all backends per league via walk-forward CV, selects the best, tunes its hyperparams with Optuna, trains final model with per-league configs
- [x] Per-league backend storage — different leagues can use different ML backends in a single saved model
- [x] **Parallel odds scraping** — concurrent game pages with retry pass for rate-limited requests (~4x faster)

### Phase 6: Dashboard & Automation
- [ ] Streamlit dashboard with daily picks display
- [ ] Cron job for automated daily scraping + prediction
- [ ] Bet tracking UI with P&L history and charts
- [ ] Line movement detection and alert system

### Phase 7: Model Improvements
- [ ] Head-to-head history features
- [ ] Pace-of-play and offensive/defensive rating features
- [ ] Ensemble with a neural net or mixture-of-experts approach
- [ ] Alternative distributions (skew-normal, empirical KDE) for non-NBA leagues

## Disclaimer

This project is for **educational and research purposes**. Sports betting involves financial risk. Always:
- Paper trade before using real money
- Never bet more than you can afford to lose
- Understand that past model performance does not guarantee future results
- Check local laws regarding sports betting in your jurisdiction

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style, and areas where help is welcome.

## License

[MIT](LICENSE)
