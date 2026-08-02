# BrainQuant Generator

A local alpha generation and evaluation pipeline for WorldQuant-style formulas.

This repository currently runs a native-only alpha compiler and evaluator pipeline, generating formulas, compiling them to Polars expressions, evaluating them against local market data, and logging results at every stage.

## Project Structure

- `main.py` - Interactive entry point. Choose between native compiler test, hybrid system, or both.
- `alpha_background_runner.py` - Main continuous execution loop that generates formulas, compiles them, evaluates metrics, and logs results.
- `local_alpha_engine.py` - AST data structures, formula generator, native compilation to Polars, and alpha metric calculation.
- `ingest.py` - Script to fetch and format market data into Parquet files.
- `operatorRAW.json` - Operator metadata used by formula generator.
- `elite_alphas.csv` - Persisted list of elite formulas that meet target metrics.
- `data/raw/` - Input Parquet files used by the engine for simulation.

## Requirements

- Python 3.11
- `polars`
- `python-dotenv`
- (Optional) Any other package imports already present in the repo, but the active pipeline depends mainly on `polars`.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install required packages:
   ```bash
   pip install polars python-dotenv
   ```

3. Place your input parquet market data under:
   ```text
   data/raw/
   ```
   Each file should be a Parquet file representing a universe snapshot.

4. Ensure the repository root contains:
   - `main.py`
   - `alpha_background_runner.py`
   - `local_alpha_engine.py`
   - `operatorRAW.json`

## Usage

Run the interactive entrypoint:

```bash
python main.py
```

Then choose:

1. `Run Native Compiler Test (single formula)`
2. `Run Hybrid Translation System (continuous)`
3. `Run Both (Native test first, then Hybrid)`
4. `Exit`

> Note: The current pipeline is configured to run native compilation only. The hybrid/LLM logic has been phased out because it was causing severe bottlenecks and invalid formula generation.

## Active Pipeline Flow

The active pipeline is in `alpha_background_runner.py`:

1. Load local data via `load_local_data()` from `local_alpha_engine.py`.
2. Generate formula ASTs with `NativeAlphaGenerator.generate_ast()`.
3. Compile ASTs to Polars `pl.Expr` objects using `compile_to_polars()`.
4. Evaluate compiled expressions against the dataset.
5. Calculate alpha metrics with `calculate_brain_metrics()`.
6. Persist logs and elite formulas.

## Logging and Output Files

The runner produces several persistent log files:

- `logs_generated.csv`
  - All formulas generated in each cycle
  - Columns: `timestamp`, `generation_id`, `formula`, `formula_length`, `translator`

- `logs_processed.csv`
  - All formulas that were successfully evaluated
  - Columns: `timestamp`, `processed_id`, `formula`, `polars_code`, `sharpe`, `turnover`, `short_pct`, `translator`

- `logs_elite.csv`
  - Elite formulas meeting target metrics
  - Columns: `timestamp`, `elite_id`, `formula`, `polars_code`, `sharpe`, `turnover`, `short_pct`, `translator`

- `elite_alphas.csv`
  - Backward-compatible elite alpha summary

- `transpiler_errors.log`
  - Detailed runtime logs and failure messages

## Metric Targets

The active evaluation targets are configured in `alpha_background_runner.py`:

- Sharpe >= `1.2`
- Turnover < `50.0%`
- Short Exposure > `40.0%`

These thresholds are defined at the top of `alpha_background_runner.py`.

## Current Known Limitations

- The system currently uses **native compilation only**.
- `rule_translator.py` and `llm_trans.py` exist in the repo but are not part of the active processing path.
- The pipeline may skip formulas with unsupported operators when `compile_to_polars()` raises `NotImplementedError`.
- Input data is expected in Parquet format under `data/raw/`.
- **Timeframe Restriction**: Data is filtered from 2017 to current date to provide a longer historical period for robust alpha evaluation. This ensures alphas are tested on recent market conditions while maintaining sufficient historical data for statistical significance.

## Troubleshooting

- If the system generates many formulas but `logs_processed.csv` remains empty, the native compiler is likely skipping unsupported AST operators.
- If the runner fails on startup, verify that `data/raw/` contains Parquet files and that `polars` is installed.
- If you see encoding errors in Windows console output, this repository no longer uses emoji characters in logs.

## Extending the Pipeline

To improve the system, consider:

- Adding support for more AST operators in `local_alpha_engine.py`.
- Improving `NativeAlphaGenerator` to generate more realistic formula structures.
- Replacing the legacy LLM/transpiler path with a faster validator or a deterministic translator.
- Adding proper tests for `compile_to_polars()` and metric calculations.

## Run Example

```bash
python main.py
```

Then select option `2` to run the continuous generator/evaluator loop. Press `Ctrl+C` to stop gracefully.

## Contact

Use this README as the central guide for understanding the current pipeline and improving the native-only alpha generation system.
