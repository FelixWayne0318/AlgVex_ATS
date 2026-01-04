# CLAUDE.md - AI Assistant Guide for AlgVex ATS

> **Version**: v10.0.4
> **Last Updated**: 2026-01-04

## Project Overview

AlgVex ATS is a cryptocurrency quantitative trading platform that integrates **Qlib** (for offline research and model training) with **Hummingbot Strategy V2** (for live trading execution). The project focuses on spot cryptocurrency trading with machine learning-based signal generation.

### Core Philosophy

```
Qlib = Offline Research & Training (no source modifications)
Hummingbot = Live Execution (Strategy V2 framework)
Unified Features = Same code path for training/backtesting/live trading
```

**Key Principle**: The training, backtesting, and live trading pipelines use the **exact same feature computation code** (`unified_features.py`) to ensure consistency.

## Repository Structure

```
AlgVex_ATS/
├── CLAUDE.md                    # This file - AI assistant guide
├── README.md                    # Project overview (Chinese)
├── requirements_tutorial.txt    # Tutorial dependencies
├── AlgVex_v10_教程_详细版.ipynb  # Main tutorial notebook
│
├── scripts/                     # Core trading scripts
│   ├── unified_features.py      # Feature computation (shared across all pipelines)
│   ├── train_model.py           # LightGBM model training
│   ├── backtest_offline.py      # Offline backtesting (same path as live)
│   ├── prepare_crypto_data.py   # Data preparation (Binance → Parquet)
│   ├── qlib_alpha_strategy.py   # V2 strategy script
│   ├── verify_integration.py    # Integration verification
│   └── generate_mock_data.py    # Mock data for testing
│
├── controllers/                 # Hummingbot V2 controllers
│   ├── __init__.py
│   └── qlib_alpha_controller.py # Main trading signal controller
│
├── conf/                        # Configuration files
│   ├── controllers/
│   │   └── qlib_alpha.yml       # Controller config
│   └── scripts/
│       └── qlib_alpha_v2.yml    # Strategy config
│
├── docs/                        # Documentation
│   ├── CORE_PLAN.md             # Core implementation plan (P0 MVP)
│   ├── EXTENSION_PLAN.md        # Extension features plan
│   ├── FUTURE_PLAN.md           # Future roadmap
│   ├── QLIB_REFERENCE.md        # Qlib integration reference
│   └── DEPLOY_WINDOWS.md        # Windows deployment guide
│
├── qlib/                        # Qlib library (submodule, unmodified)
│   ├── qlib/                    # Core Qlib code
│   ├── examples/                # Qlib examples
│   ├── tests/                   # Qlib tests
│   └── pyproject.toml           # Qlib dependencies
│
├── hummingbot/                  # Hummingbot framework (submodule)
│   ├── hummingbot/              # Core Hummingbot code
│   ├── controllers/             # Built-in controllers
│   ├── scripts/                 # Built-in scripts
│   ├── conf/                    # Configuration templates
│   └── pyproject.toml           # Hummingbot dependencies
│
├── oss_eval/                    # Evaluation reports
│   ├── BINANCE_DATA_INTEGRATION_GUIDE.md
│   ├── BINANCE_PERPETUAL_EVALUATION_REPORT.md
│   └── CRYPTO_SPECIFIC_DATA_GUIDE.md
│
└── .github/
    └── workflows/               # CI/CD workflows
        ├── ci.yml               # Basic CI checks
        ├── integration-test.yml # Integration tests
        ├── claude.yml           # Claude AI workflow
        └── pr-review.yml        # PR review automation
```

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AlgVex Data Flow                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TRAINING PIPELINE:                                                  │
│    Binance API → Parquet → unified_features → normalizer.fit        │
│                                    ↓                                 │
│                               LightGBM train → lgb_model.txt         │
│                                                                      │
│  BACKTEST PIPELINE:                                                  │
│    Parquet → unified_features → normalizer.transform → predict       │
│                                    ↓                                 │
│                            simulate_trading → metrics                 │
│                                                                      │
│  LIVE PIPELINE:                                                      │
│    Hummingbot candles → unified_features → normalizer.transform      │
│                                    ↓                                 │
│                  QlibAlphaController → PositionExecutor              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **unified_features.py** | `scripts/` | Feature computation (59 features including KBAR, ROC, MA, STD, etc.) |
| **FeatureNormalizer** | `scripts/unified_features.py` | Z-score normalization with train-time statistics |
| **QlibAlphaController** | `controllers/` | Hummingbot V2 controller for signal generation |
| **train_model.py** | `scripts/` | LightGBM training with time-series split |
| **backtest_offline.py** | `scripts/` | Offline backtesting with fee/slippage simulation |

### Model Artifacts

Models are saved to `~/.algvex/models/qlib_alpha/`:

```
~/.algvex/models/qlib_alpha/
├── lgb_model.txt         # LightGBM model
├── normalizer.pkl        # Normalization parameters (mean, std)
├── feature_columns.pkl   # Feature column order
└── metadata.json         # Training metadata
```

## Development Workflow

### 1. Data Preparation

```bash
# Download and convert Binance data to Parquet format
python scripts/prepare_crypto_data.py --instruments btcusdt ethusdt --freq 1h
```

Data is stored in `~/.algvex/data/1h/`:
- `btcusdt.parquet`
- `ethusdt.parquet`
- `metadata.json`

### 2. Model Training

```bash
python scripts/train_model.py \
    --instruments btcusdt ethusdt \
    --train-start 2023-01-01 \
    --train-end 2024-06-15 \
    --valid-start 2024-07-01 \
    --valid-end 2024-12-31
```

**Important**: Leave a 2-week gap between train-end and valid-start to prevent data leakage.

### 3. Offline Backtesting

```bash
python scripts/backtest_offline.py \
    --instruments btcusdt \
    --test-start 2024-07-01 \
    --signal-threshold 0.001 \
    --stop-loss 0.02 \
    --take-profit 0.03
```

### 4. Live Trading (Paper/Real)

```bash
# Start Hummingbot with the V2 strategy
cd hummingbot
./start
# In Hummingbot: start --script qlib_alpha_strategy.py
```

## Coding Conventions

### Python Style

- **Python Version**: 3.8 - 3.12 supported
- **Line Length**: 120 characters (Black formatter)
- **Import Order**: Standard library → Third-party → Local (isort)
- **Type Hints**: Use when practical, especially for public APIs

### Feature Computation

The 59 features in `FEATURE_COLUMNS` must remain consistent across all pipelines:

```python
FEATURE_COLUMNS = [
    # KBAR (9): KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2
    # ROC (5): ROC5, ROC10, ROC20, ROC30, ROC60
    # MA (5): MA5, MA10, MA20, MA30, MA60
    # STD (5): STD5, STD10, STD20, STD30, STD60
    # MAX (5): MAX5, MAX10, MAX20, MAX30, MAX60
    # MIN (5): MIN5, MIN10, MIN20, MIN30, MIN60
    # QTLU (5): QTLU5, QTLU10, QTLU20, QTLU30, QTLU60
    # QTLD (5): QTLD5, QTLD10, QTLD20, QTLD30, QTLD60
    # RSV (5): RSV5, RSV10, RSV20, RSV30, RSV60
    # CORR (5): CORR5, CORR10, CORR20, CORR30, CORR60
    # CORD (5): CORD5, CORD10, CORD20, CORD30, CORD60
]
```

### Configuration Files

- Use YAML for controller/strategy configs
- Use Decimal for monetary values (not float)
- All time values in UTC

### Error Handling

- Use `strict=True` for normalizer in live/backtest (fail on missing features)
- Use `strict=False` only during training/debugging
- Log errors with `self.logger.error()` in controllers

## Testing

### CI Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push/PR to main | Basic checks, YAML lint |
| `integration-test.yml` | Push/PR with path filters | Qlib/Hummingbot integration |
| `pr-review.yml` | PR events | Automated code review |

### Running Tests Locally

```bash
# Check Python syntax for all scripts
python -m py_compile scripts/*.py

# YAML lint
yamllint -d relaxed .github/workflows/*.yml

# Integration verification
python scripts/verify_integration.py
```

### Acceptance Criteria (Backtest)

- Sharpe Ratio > 0.5
- Max Drawdown < 30%
- Win Rate > 40%

## Configuration Reference

### Controller Config (`conf/controllers/qlib_alpha.yml`)

```yaml
id: qlib_alpha_btc              # Unique identifier
controller_name: qlib_alpha
controller_type: directional_trading

connector_name: binance
trading_pair: BTC-USDT
order_amount_usd: 100

model_dir: ~/.algvex/models/qlib_alpha

signal_threshold: 0.005         # 0.5% return threshold
stop_loss: 0.02                 # 2%
take_profit: 0.03               # 3%
time_limit: 3600                # 1 hour
cooldown_interval: 60           # 60 seconds
```

### Strategy Config (`conf/scripts/qlib_alpha_v2.yml`)

See `docs/CORE_PLAN.md` for full configuration reference.

## Important Notes for AI Assistants

### DO

1. **Maintain Feature Consistency**: Any changes to feature computation in `unified_features.py` MUST be reflected in training, backtesting, AND live trading.

2. **Use Parquet for Data**: All offline data should use Parquet format, not Qlib's `.bin` format.

3. **Respect Zero-Modification Policy**: Do NOT modify Qlib or Hummingbot source code. Use runtime configuration or extension points.

4. **Use Decimal for Money**: Always use `Decimal` type for monetary calculations to avoid floating-point precision issues.

5. **Test with `strict=True`**: When testing the normalizer, use `strict=True` to catch feature mismatches early.

6. **Document in Chinese and English**: The project uses both languages; maintain consistency with existing patterns.

### DON'T

1. **Don't Use Alpha158/DatasetH**: These Qlib components depend on `.bin` data providers, which conflict with the Parquet-based architecture.

2. **Don't Modify FEATURE_COLUMNS Order**: The order is critical for model predictions. Changes require retraining.

3. **Don't Skip the Validation Gap**: Always leave a 1-2 week gap between training and validation periods.

4. **Don't Use `iloc[-1]` for Live Signals**: Use `iloc[-2]` (the last closed bar) to match backtest behavior.

5. **Don't Commit Secrets**: Never commit API keys, `.env` files, or credentials.

### Common Tasks

#### Adding a New Feature

1. Add to `FEATURE_COLUMNS` in `unified_features.py`
2. Implement computation in `compute_unified_features()`
3. Retrain the model
4. Update `feature_columns.pkl`

#### Changing Signal Threshold

1. Update `conf/controllers/qlib_alpha.yml`
2. Run backtest with new threshold to validate
3. Deploy to paper trading first

#### Debugging Feature Mismatch

```python
# Check feature alignment
from scripts.unified_features import FEATURE_COLUMNS
import pickle

with open("~/.algvex/models/qlib_alpha/feature_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

print("Code columns:", FEATURE_COLUMNS)
print("Model columns:", model_columns)
print("Match:", FEATURE_COLUMNS == model_columns)
```

## Dependencies

### Core Dependencies

```
pandas>=1.5.0
numpy>=1.21.0
lightgbm>=3.3.0
pyyaml>=6.0
requests>=2.28.0
```

### Qlib Dependencies

See `qlib/pyproject.toml` - requires Python 3.8+, NumPy, Pandas, LightGBM, MLflow.

### Hummingbot Dependencies

See `hummingbot/pyproject.toml` - requires Python 3.10+, NumPy 2.x, Cython.

## Resources

- **Core Plan**: `docs/CORE_PLAN.md` - Detailed implementation specification
- **Extension Plan**: `docs/EXTENSION_PLAN.md` - Future features
- **Qlib Reference**: `docs/QLIB_REFERENCE.md` - Qlib integration notes
- **Tutorial Notebook**: `AlgVex_v10_教程_详细版.ipynb` - Step-by-step guide

## Contact & Contributing

See `.github/CONTRIBUTING.md` in the Hummingbot submodule for contribution guidelines.

For issues specific to AlgVex integration, file issues in this repository.
