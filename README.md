# BrainQuant Generator

A high-performance, local alpha formula generation, compilation, and evaluation pipeline for quantitative trading (WorldQuant-style expressions).

The core engine native-compiles abstract syntax tree (AST) alpha formulas directly into **Polars (`pl.Expr`) vector expressions**, evaluates backtest performance metrics against local Parquet market data snapshot files, and filters elite signal candidates into persistent logs.

---

## 🏗 System Architecture & Module Details

Here is an exact breakdown of what each module does and where specific logic lives:

```
                  ┌───────────────────────┐
                  │       main.py         │ (CLI Menu Entrypoint)
                  └───────────┬───────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
  ┌─────────────────────────┐   ┌───────────────────────────┐
  │  local_alpha_engine.py  │   │alpha_background_runner.py │
  │ (AST Gen & Polars Comp) │   │ (Continuous Loop & Logs)  │
  └────────────┬────────────┘   └─────────────┬─────────────┘
               │                              │
               ├──────────────────────────────┘
               ▼
  ┌─────────────────────────┐
  │  data/raw/*.parquet     │ (Market Data Ingestion by ingest.py)
  └─────────────────────────┘
```

---

### 1. `main.py` (User Interface Entrypoint)
- **Role**: Command-Line Interface (CLI) launcher.
- **Functionality**:
  - Displays interactive menu options (`1`: Single AST test compile, `2`: Continuous background run loop, `3`: Combined).
  - Uses `subprocess.run` to invoke `local_alpha_engine.py` or `alpha_background_runner.py` in isolated execution context.

---

### 2. `local_alpha_engine.py` (Core Engine, AST, Compiler & Backtest)
- **Role**: Heart of the expression generator, AST parser, Polars compiler, and performance evaluator.
- **Key Components**:
  - `ASTNode`: Data structure representing nodes in the AST (`type`: `field`, `constant`, or `operator`). Contains `to_string()` method to format human-readable WorldQuant formulas.
  - `load_local_data(data_dir="data/raw")`:
    - Scans all `.parquet` files under `data/raw/`.
    - Automatically maps & normalizes standard price/volume schema (`date`, `open`, `high`, `low`, `close`, `volume`, `vwap`).
    - Standardizes date strings into datetime objects and applies date range filters (2017 to current date).
    - Sorts by ticker & date, returning a unified `pl.LazyFrame`.
  - `NativeAlphaGenerator`:
    - Uses `operatorRAW.json` metadata to construct random, valid AST trees with defined recursive depth bounds (`max_depth`).
    - Supports operators (unary, binary, rolling time-series like `ts_mean`, `ts_std`, `rank`, `delay`, `delta`).
  - `compile_to_polars(node: ASTNode)`:
    - Recursively parses `ASTNode` hierarchies and translates them directly into executable `polars.Expr` statements.
    - Handles field column mapping, math operations (`+`, `-`, `*`, `/`, `abs`, `log`, `sqrt`), and time-series rolling functions windowed over ticker partitions (`pl.col(...).over("ticker")`).
  - `calculate_brain_metrics(df: pl.DataFrame, signal_col="alpha_signal")`:
    - Executes vector cross-sectional signals.
    - Neutralizes alphas cross-sectionally by subtracting group mean (`pl.col(signal_col) - pl.col(signal_col).mean().over("date")`).
    - Computes 1-day forward returns per asset (`close.shift(-1) / close - 1`).
    - Calculates portfolio metrics:
      - **Sharpe Ratio**: Annualized ratio of portfolio mean return over return volatility (\(\text{Sharpe} = \sqrt{252} \cdot \frac{\mu}{\sigma}\)).
      - **Turnover**: Average daily portfolio position change rate.
      - **Short Percentage**: Fraction of total allocation weight in short positions.

---

### 3. `alpha_background_runner.py` (Continuous Backtest Runner & Logging Pipeline)
- **Role**: Execution orchestrator that continuously generates, backtests, filters, and logs alphas.
- **Functionality**:
  - Imports engine functions from `local_alpha_engine.py`.
  - Runs continuous generation loops in configurable batch sizes (`BATCH_SIZE = 3`).
  - Evaluates performance metrics against target criteria:
    - **Sharpe Target**: \(\ge 1.0\) (or configured threshold)
    - **Turnover Max**: \(\le 50.0\%\)
    - **Short Min**: \(\ge 40.0\%\)
  - Logs results to CSV files (`logs_generated.csv`, `logs_processed.csv`, `logs_elite.csv`, `elite_alphas.csv`).
  - Writes failure tracebacks and syntax/compilation issues to `transpiler_errors.log`.

---

### 4. `ingest.py` (Market Data Acquisition & Ingestion)
- **Role**: Utility script for downloading raw stock price data and saving snapshot Parquet files.
- **Functionality**:
  - Uses `pytickersymbols` and `yfinance` to pull historical OHLCV data for index constituents (S&P 500, NASDAQ 100, DAX, CAC 40, SMI, AEX).
  - Multi-threaded download manager using Python `ThreadPoolExecutor`.
  - Normalizes column names into canonical format (`date`, `open`, `high`, `low`, `close`, `volume`, `adj_close`).
  - Saves formatted output as individual Parquet files into `data/raw/<ticker>.parquet`.

---

### 5. `operatorRAW.json` (Operator Schema Definition)
- **Role**: JSON schema defining operator signatures used by `NativeAlphaGenerator`.
- **Contains**:
  - Operator types (time-series, cross-sectional, element-wise arithmetic).
  - Parameter expectations (e.g., number of input children nodes, lookback window constraints).

---

### 6. Legacy Modules (Phrased out/Deprecated)
- `rule_translator.py`: Legacy rule-based string translator (bypassed in favor of direct AST -> Polars compiler).
- `llm_trans.py`: Legacy LLM prompt transpiler (bypassed to remove rate-limiting latency and code translation errors).

---

## 📊 Data & File Outputs

The pipeline generates and ignores (via `.gitignore`) the following artifacts during runtime:

| File / Path | Description |
| :--- | :--- |
| `data/raw/*.parquet` | Market dataset snapshots per ticker. |
| `logs_generated.csv` | Log of all AST formulas generated per cycle. |
| `logs_processed.csv` | Log of valid formulas successfully evaluated with Sharpe, turnover, & short metrics. |
| `logs_elite.csv` | High-performing formulas passing Sharpe, turnover, and short constraints. |
| `elite_alphas.csv` | Summary export file of elite signals. |
| `transpiler_errors.log` | Stack traces and uncompiled operator error details. |

---

## 🚀 Quickstart Guide

### 1. Requirements
- Python 3.11+
- Installed packages: `polars`, `python-dotenv`, `yfinance`, `pytickersymbols`, `pandas`, `tqdm`

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/DheerajS-DM/alpha-generator.git
cd alpha-generator

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows PowerShell

# Install dependencies
pip install polars python-dotenv yfinance pytickersymbols pandas tqdm
```

### 3. Ingest Market Data
Run `ingest.py` to download market data into `data/raw/`:
```bash
python ingest.py
```

### 4. Run Alpha Generator Engine
Launch the CLI launcher:
```bash
python main.py
```
Select **Option 2** for continuous alpha discovery and evaluation.

---

## ⚙️ Configuration & Thresholds

To adjust backtest filters or generator parameters, edit top-level constants in `alpha_background_runner.py`:

```python
SHARPE_THRESHOLD = 1.0  # Minimum Sharpe ratio target
TURNOVER_MAX = 50.0      # Maximum allowed turnover percentage
SHORT_MIN = 40.0         # Minimum required short exposure percentage
BATCH_SIZE = 3           # Formulations per processing batch
```
