# BrainQuant Generator

A high-performance, local alpha formula generation, compilation, and evaluation pipeline for quantitative trading (WorldQuant-style expressions).

The core engine native-compiles abstract syntax tree (AST) alpha formulas directly into Polars (`pl.Expr`) vector expressions, evaluates backtest performance metrics against local Parquet market data snapshot files, and filters elite signal candidates into persistent logs.

---

## Internal Dynamics & Module Breakdown

This section details the exact internal workflow and execution dynamics for each `.py` file in the project.

---

### 1. `main.py` (CLI Launcher & Process Initiator)

#### Role & Purpose
`main.py` serves as the user-facing entrypoint for launching system operations. It presents a simple text-based command-line interface (CLI) to select the execution mode.

#### Internal Execution Flow
1. **User Interaction**: Prompts the user via stdin to choose an execution path:
   - Choice 1: Single test compilation run via `local_alpha_engine.py`.
   - Choice 2: Continuous background processing loop via `alpha_background_runner.py`.
   - Choice 3: Combined execution (single test run followed by continuous processing).
   - Choice 4: Exit.
2. **Subprocess Isolation**: Executes child processes using `subprocess.run([sys.executable, script_name])` with `cwd` set to the script directory. This guarantees clean memory management and prevents module state corruption across runs.
3. **Interrupt Handling**: Listens for `KeyboardInterrupt` (Ctrl+C) to terminate running subprocesses gracefully without leaving zombie Python instances.

---

### 2. `local_alpha_engine.py` (AST Engine, Polars Compiler & Metrics Evaluator)

#### Role & Purpose
`local_alpha_engine.py` is the core mathematical and computational engine. It defines AST data structures, constructs random formula trees, compiles AST nodes directly to Polars vector expressions, loads price/volume market datasets, and calculates quantitative signal performance metrics.

#### Internal Dynamics by Component

#### A. Data Structure (`ASTNode`)
- Dataclass holding AST state:
  - `type`: Node category (`field`, `constant`, or `operator`).
  - `value`: Node identifier (e.g. `close`, `ts_mean`, `0.5`).
  - `children`: List of nested `ASTNode` objects.
- `to_string()` method recursively walks the AST hierarchy to produce human-readable WorldQuant-style expressions (e.g. `ts_mean(rank(close), 10)`).

#### B. Market Data Ingestion (`load_local_data`)
- Scans `data/raw/*.parquet` for asset snapshots.
- Extracts ticker symbol from file names and scans schema names.
- Maps variations of column names to canonical schema: `date`, `open`, `high`, `low`, `close`, `volume`, `vwap`.
- Filters data historical range (2017 to present day).
- Returns a unified `pl.LazyFrame` partitioned by asset and date.

#### C. AST Formula Generator (`NativeAlphaGenerator`)
- Parses `operatorRAW.json` to extract operator definitions, parameter specs, and lookback windows.
- `generate_ast(depth, max_depth)` recursively generates expressions:
  - Terminal depth (`depth == max_depth`): Randomly picks a dataset field or constant scalar.
  - Non-terminal depth (`depth < max_depth`): Randomly selects an operator from schema rules, instantiates required child nodes, and enforces integer parameter constraints (e.g. lookback window between 5 and 60 days).

#### D. Vector Compiler (`compile_to_polars`)
- Recursively parses `ASTNode` hierarchies and translates them directly into executable `pl.Expr` trees.
- Operator mappings:
  - **Arithmetic & Element-wise**: `add`, `subtract`, `multiply`, `divide`, `abs`, `log`, `sqrt`, `sign`.
  - **Cross-Sectional**: `rank` (`pl.col().rank() / count()`), `zscore` (`(x - mean) / std`), `normalize` (`x / sum(|x|)`).
  - **Time-Series Rolling (windowed over `.over("ticker")`)**:
    - `ts_mean` -> `pl.col().rolling_mean(window)`
    - `ts_std` -> `pl.col().rolling_std(window)`
    - `ts_min` / `ts_max` -> `pl.col().rolling_min(window)` / `rolling_max()`
    - `ts_delay` -> `pl.col().shift(window)`
    - `ts_delta` -> `pl.col() - pl.col().shift(window)`
    - `ts_corr` -> `pl.rolling_corr(x, y, window)`
    - `ts_covariance` -> `pl.rolling_cov(x, y, window)`

#### E. Backtest & Portfolio Metrics (`calculate_brain_metrics`)
1. Appends compiled expression to market data frame as `alpha_signal`.
2. Computes cross-sectional mean-neutralized positions:
   $$\text{Signal}_{\text{neutralized}} = \text{Signal} - \text{Mean}(\text{Signal})_{\text{date}}$$
3. Computes 1-day forward asset returns:
   $$R_{i, t+1} = \frac{\text{Close}_{i, t+1}}{\text{Close}_{i, t}} - 1$$
4. Evaluates daily portfolio returns across universe:
   $$R_p = \sum (\text{Signal}_{\text{weights}} \times R_{i, t+1})$$
5. Calculates final metrics:
   - **Sharpe Ratio**: Annualized risk-adjusted return ($\sqrt{252} \cdot \frac{\mu_{R_p}}{\sigma_{R_p}}$).
   - **Turnover**: Daily percentage shift in portfolio allocation weights ($\frac{1}{N} \sum |\text{Weight}_t - \text{Weight}_{t-1}| \times 100$).
   - **Short Percentage**: Proportion of negative allocation weights relative to gross exposure.

---

### 3. `alpha_background_runner.py` (Continuous Batch Processing Loop & Logger)

#### Role & Purpose
`alpha_background_runner.py` acts as the production task driver. It executes continuous iteration loops to generate batches of formula ASTs, compile them via `local_alpha_engine.py`, run vector evaluations, evaluate filter thresholds, and log outputs to disk.

#### Internal Execution Dynamics
1. **Initialization**:
   - Loads `.env` configuration file via `python-dotenv`.
   - Initializes Dual Logging: Writes execution logs to stdout and appends stack traces to `transpiler_errors.log`.
   - Calls `load_local_data()` once on startup to cache lazy evaluation data into memory (`pl.LazyFrame`).
2. **Batch Iteration Loop**:
   - Loops indefinitely (or until user stops process).
   - In each iteration:
     a. **Generation**: Invokes `NativeAlphaGenerator.generate_ast()` to create `BATCH_SIZE` AST formulas. Appends metadata to `logs_generated.csv`.
     b. **Compilation & Execution**: Passes each formula to `compile_to_polars()`. If compilation succeeds, evaluates the Polars expression against market data.
     c. **Metric Evaluation**: Passes signal data to `calculate_brain_metrics()`. If processing succeeds, appends stats to `logs_processed.csv`.
     d. **Elite Filtering**: Checks metrics against configurable thresholds:
        - Sharpe Ratio $\ge$ `SHARPE_THRESHOLD` (Default: `1.0`)
        - Turnover $\le$ `TURNOVER_MAX` (Default: `50.0%`)
        - Short Exposure $\ge$ `SHORT_MIN` (Default: `40.0%`)
     e. **Logging**: If criteria are satisfied, saves the elite formula and Polars snippet into `logs_elite.csv` and `elite_alphas.csv`.
     f. **Error Catching**: Traps compilation and execution exceptions (`NotImplementedError`, `PolarsPanicError`, `ZeroDivisionError`), logging tracebacks to `transpiler_errors.log` without crashing the main loop.

---

### 4. `ingest.py` (Market Data Acquisition Utility)

#### Role & Purpose
`ingest.py` is an independent utility module used to download historical OHLCV pricing data for global stock indices and store formatted snapshot files in Parquet format.

#### Internal Execution Dynamics
1. **Target Universe Setup**:
   - Queries `pytickersymbols` to pull constituent lists for global indices (S&P 500, NASDAQ 100, DAX, CAC 40, SMI, AEX).
2. **Concurrent Download Driver**:
   - Uses `concurrent.futures.ThreadPoolExecutor(max_workers=8)` to fetch ticker historical price data asynchronously from Yahoo Finance (`yfinance`).
   - Retries failed downloads up to `RETRY_LIMIT` (3 attempts) with exponential backoff delays.
3. **Data Normalization & Formatting**:
   - Standardizes schema column names (`date`, `open`, `high`, `low`, `close`, `volume`, `adj_close`).
   - Converts datatypes to strict 64-bit floating point numbers and ISO datetime formats.
4. **Storage Output**:
   - Writes individual ticker files as binary compressed Parquet snapshots under `data/raw/<ticker>.parquet`.

---

## Summary Table of Script Inputs and Outputs

| Script | Primary Inputs | Primary Outputs / Functions |
| :--- | :--- | :--- |
| `main.py` | User CLI selection (1-4) | Spawns `local_alpha_engine.py` or `alpha_background_runner.py` subprocesses |
| `local_alpha_engine.py` | `operatorRAW.json`, `data/raw/*.parquet` | Builds ASTs, compiles Polars `pl.Expr`, computes Sharpe/Turnover/Short metrics |
| `alpha_background_runner.py` | Output of `local_alpha_engine.py` | Continuously runs backtests, writes to `logs_*.csv` & `transpiler_errors.log` |
| `ingest.py` | Yahoo Finance API, index constituent lists | Downloads price history and writes `data/raw/<ticker>.parquet` files |

---

## Data & File Outputs

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

## Quickstart Guide

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

## Configuration & Thresholds

To adjust backtest filters or generator parameters, edit top-level constants in `alpha_background_runner.py`:

```python
SHARPE_THRESHOLD = 1.0  # Minimum Sharpe ratio target
TURNOVER_MAX = 50.0      # Maximum allowed turnover percentage
SHORT_MIN = 40.0         # Minimum required short exposure percentage
BATCH_SIZE = 3           # Formulations per processing batch
```
