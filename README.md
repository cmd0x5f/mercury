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
NBA Game Data → Feature Engineering → XGBoost Prediction → Folded Normal Distribution → Bucket Probabilities
                                                                                              ↓
SportsPlus Odds → Implied Probabilities ──────────────────────────────────→ Edge Calculation → +EV Picks
```

1. **Predict the signed margin** (e.g., "home team wins by 5") using an XGBoost model trained on NBA + 30 international leagues
2. **Convert to a probability distribution** over absolute margin using a [folded normal distribution](https://en.wikipedia.org/wiki/Folded_normal_distribution) with per-league σ — the key insight is that |N(μ, σ)| naturally models "any team wins by X"
3. **Scrape bookmaker odds** from SportsPlus.ph for the "Any Team Winning Margin" market
4. **Find value** where model probability exceeds implied probability by a configurable threshold
5. **Size bets** using 1/4 Kelly criterion with exposure caps

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

# Step 1b (optional): Collect international league data from Flashscore
make collect-leagues

# Step 2: Train the model (NBA-only or all leagues)
make train

# Step 3: Scrape today's odds from SportsPlus (~10 min for full slate)
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
  train            Train the margin prediction model
  evaluate         Run walk-forward backtesting
  scrape           Scrape winning margin odds from SportsPlus
  picks            Show +EV picks with Kelly-sized stakes
  status           Show bet tracking P&L summary

Options for train/evaluate:
  -l, --league    Train/evaluate on single league (default: all)

Options for picks:
  -b, --bankroll  Current bankroll (default: 10000)
  -e, --min-edge  Minimum edge threshold (default: 0.05 = 5%)
  -k, --kelly     Kelly fraction (default: 0.25 = quarter Kelly)
  -d, --date      Game date YYYY-MM-DD (default: latest with odds)
```

## Project Structure

```
sportsbetting/
├── src/
│   ├── data/
│   │   ├── data_store.py           # SQLite storage for games, odds, bets
│   │   ├── nba_collector.py        # NBA stats API data collection
│   │   ├── flashscore_scraper.py   # Playwright scraper for international league results
│   │   ├── sportsplus_scraper.py   # Playwright scraper for bookmaker odds
│   │   ├── league_matcher.py       # Fuzzy league + team name matching (rapidfuzz)
│   │   └── team_names.py           # NBA team name mapping
│   ├── features/
│   │   ├── team_strength.py        # Elo rating system (per-league)
│   │   ├── form.py                 # Rolling margin & scoring averages
│   │   ├── context.py              # Rest days, back-to-backs, schedule position
│   │   └── builder.py              # Combines all features + league_id encoding
│   ├── model/
│   │   ├── margin_model.py         # XGBoost regressor with per-league σ
│   │   ├── distribution.py         # Folded normal → bucket probabilities
│   │   └── calibration.py          # Platt scaling (Phase 3)
│   ├── betting/
│   │   ├── value_calculator.py     # Edge = model_prob - implied_prob
│   │   ├── kelly.py                # 1/4 Kelly criterion stake sizing
│   │   └── tracker.py              # Bet recording and P&L tracking
│   └── app/
│       └── cli.py                  # Click CLI entry point
├── config/
│   ├── settings.yaml               # Model + betting configuration
│   └── league_mappings.yaml        # 30+ league → Flashscore URL mappings
├── tests/                          # 107 unit tests + 6 live integration tests
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
| [Flashscore](https://flashscore.com) (Playwright scraper) | 30+ international basketball leagues | Free |
| SportsPlus.ph (Playwright scraper) | Winning margin odds for upcoming games | Free |
| [API-Basketball](https://rapidapi.com/api-sports/api/api-basketball) (future) | 115+ leagues via REST API | $7.99/mo |

## Testing

```bash
make test           # 107 unit tests, ~5 seconds
make test-live      # 6 live integration tests (hits Flashscore, ~60s)
make test-cov       # with coverage report
make lint           # ruff linting
```

Tests cover: data store CRUD, Elo math properties, distribution sum-to-one invariants, Kelly boundary conditions, margin-to-bucket mapping, model train/predict/save/load, team name mapping, scraper date parsing, bucket regex, and **live DOM selector validation** against Flashscore (catches site structure changes).

## Roadmap

### ~~Phase 2: International Leagues~~ ✅
- [x] Flashscore scraper for 30+ leagues (Euroleague, ACB, BBL, LNB, CBA, KBL, etc.)
- [x] League + team name fuzzy matching between SportsPlus ↔ Flashscore via `rapidfuzz`
- [x] `league_id` as categorical feature with per-league σ estimation
- [x] Live integration tests to detect Flashscore DOM changes
- [ ] SofaScore fallback when Flashscore scraping fails

### Phase 3: Backtesting & Calibration
- [ ] Walk-forward backtesting framework with proper train/test split
- [ ] Platt scaling calibration for bucket probabilities
- [ ] Backtest reports: hit rate per bucket, ROI curve, max drawdown
- [ ] Tune edge threshold (3% / 5% / 7% / 10%) vs number of bets vs ROI

### Phase 4: Dashboard & Automation
- [ ] Streamlit dashboard with daily picks display
- [ ] Cron job for automated daily scraping + prediction
- [ ] Bet tracking UI with P&L history and charts
- [ ] Line movement detection and alert system

### Phase 5: Model Improvements
- [ ] Add player-level features (injuries, rest, minutes load)
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
