import time
import csv
import os
import logging
import ast
from datetime import datetime
import polars as pl
from dotenv import load_dotenv

load_dotenv() 

# Import your data loader and metrics from the engine
from local_alpha_engine import AlphaGenerator, load_local_data, calculate_rolling_metrics, compile_to_polars
from operator_registry import OperatorRegistry
from alpha_validator import validate_ast

# --- Configuration ---
SHARPE_THRESHOLD = 1.0
WORST_SHARPE_MIN = 0.0
CONSISTENCY_MIN = 60.0
TURNOVER_MAX = 50.0
SHORT_MIN = 40.0
BATCH_SIZE = 5

BEST_ALPHAS_FILE = "elite_alphas.csv"
GENERATED_LOG_FILE = "logs_generated.csv"
PROCESSED_LOG_FILE = "logs_processed.csv"
ELITE_LOG_FILE = "logs_elite.csv"
ERROR_LOG_FILE = "transpiler_errors.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ERROR_LOG_FILE),
        logging.StreamHandler()
    ]
)

class AlphaLogger:
    """Logs alphas at each stage: generated, processed, elite"""
    def __init__(self):
        self.generated_count = 0
        self.processed_count = 0
        self.elite_count = 0
    
    def log_generated(self, formula):
        self.generated_count += 1
        file_exists = os.path.isfile(GENERATED_LOG_FILE)
        with open(GENERATED_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "generation_id", "formula"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                self.generated_count,
                formula
            ])
    
    def log_processed(self, formula, metrics, success=True, error=None):
        if success:
            self.processed_count += 1
            file_exists = os.path.isfile(PROCESSED_LOG_FILE)
            with open(PROCESSED_LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "timestamp", "processed_id", "formula", 
                        "median_sharpe", "worst_sharpe", "consistency", "turnover", "short_pct", "windows"
                    ])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    self.processed_count,
                    formula,
                    round(metrics['median_sharpe'], 4),
                    round(metrics['worst_sharpe'], 4),
                    round(metrics['consistency'], 2),
                    round(metrics['mean_turnover'], 2),
                    round(metrics['mean_short_pct'], 2),
                    metrics['windows']
                ])
            logging.info(f"[OK] [PROC #{self.processed_count}] MedSharpe={metrics['median_sharpe']:.2f} | Cons={metrics['consistency']:.0f}% | Turn={metrics['mean_turnover']:.1f}%")
        else:
            logging.warning(f"[FAIL] [PROC] {error}")
    
    def log_elite(self, formula, metrics):
        self.elite_count += 1
        file_exists = os.path.isfile(ELITE_LOG_FILE)
        with open(ELITE_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "elite_id", "formula", 
                    "median_sharpe", "worst_sharpe", "consistency", "turnover", "short_pct", "windows"
                ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                self.elite_count,
                formula,
                round(metrics['median_sharpe'], 4),
                round(metrics['worst_sharpe'], 4),
                round(metrics['consistency'], 2),
                round(metrics['mean_turnover'], 2),
                round(metrics['mean_short_pct'], 2),
                metrics['windows']
            ])
        
        file_exists = os.path.isfile(BEST_ALPHAS_FILE)
        with open(BEST_ALPHAS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "sharpe", "worst_sharpe", "consistency", "turnover", "short_pct", "formula"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                round(metrics['median_sharpe'], 4),
                round(metrics['worst_sharpe'], 4),
                round(metrics['consistency'], 2),
                round(metrics['mean_turnover'], 2),
                round(metrics['mean_short_pct'], 2),
                formula
            ])
        
        logging.info(f"[ELITE #{self.elite_count}] MedSharpe={metrics['median_sharpe']:.2f} | Turn={metrics['mean_turnover']:.1f}% | Short={metrics['mean_short_pct']:.1f}%")

alpha_logger = AlphaLogger()

def run_diagnostics(registry: OperatorRegistry, df_lazy: pl.LazyFrame):
    print("\n" + "="*58)
    print("  BRAINQUANT GENERATOR — STARTUP DIAGNOSTICS")
    print("="*58)
    print("OPERATOR REGISTRY:")
    print(f"  Total operators in JSON: {len(registry.get_all())}")
    print(f"  Beginner-accessible (REGULAR): {len(registry.get_beginner_usable())}")
    print(f"  Implemented in Polars (usable): {len(registry.get_implementable())}")
    
    print("\nROLLING WINDOW CONFIG:")
    print("  Window sizes: 1-year (252d), 2-year (504d)")
    print("  Step size: 63 days (quarterly)")
    print("  Delay: 1 (applied to final signal)")
    
    print("\nELITE THRESHOLDS:")
    print(f"  Median Sharpe >= {SHARPE_THRESHOLD}")
    print(f"  Worst Sharpe  >  {WORST_SHARPE_MIN}")
    print(f"  Consistency   >= {CONSISTENCY_MIN}%")
    print(f"  Turnover      <  {TURNOVER_MAX}%")
    print(f"  Short %       >  {SHORT_MIN}%")
    print("="*58 + "\n")

def main():
    try:
        logging.info("Initializing engine...")

        # Initialize Registry
        registry = OperatorRegistry("operatorRAW.json")
        
        # Initialize Data
        df_lazy = load_local_data()
        try:
            df_eager = df_lazy.collect()
            logging.info(f"Data loaded into memory. Total rows: {df_eager.height}")
        except Exception as e:
            logging.warning(f"Could not collect lazy frame into memory: {e}. Using lazy evaluation.")
            df_eager = df_lazy
            
        local_fields = ["open", "high", "low", "close", "volume"]
        generator = AlphaGenerator(registry, valid_fields=local_fields)
        
        run_diagnostics(registry, df_lazy)
        
    except Exception as e:
        logging.error(f"Initialization failed: {e}", exc_info=True)
        return

    logging.info("Starting Batch Compilation Loop...")
    found_count = 0
    cycle = 1
    
    try:
        while True: 
            logging.info(f"--- Cycle {cycle} ---")
            
            try:
                # Generate and validate batch of ASTs
                batch_asts = []
                batch_formulas = []
                attempts = 0
                while len(batch_asts) < BATCH_SIZE and attempts < BATCH_SIZE * 5:
                    attempts += 1
                    ast_node = generator.generate_ast()
                    is_valid, errors = validate_ast(ast_node, registry, local_fields)
                    if is_valid:
                        batch_asts.append(ast_node)
                        formula_str = ast_node.to_string()
                        batch_formulas.append(formula_str)
                        alpha_logger.log_generated(formula_str)
                    else:
                        logging.debug(f"AST Validation Failed: {errors[0]}")
                        
                if not batch_asts:
                    logging.warning("Could not generate valid ASTs. Check operator registry.")
                    time.sleep(1)
                    continue

                compiled_batch = {}
                for i, ast_node in enumerate(batch_asts):
                    try:
                        polars_expr = compile_to_polars(ast_node)
                        compiled_batch[batch_formulas[i]] = polars_expr
                    except Exception as e:
                        logging.warning(f"  [FAIL] Compiler failed for {batch_formulas[i][:60]}... - {e}")
                
                if not compiled_batch:
                    logging.warning("Compiler failed on all formulas. Skipping cycle.")
                    cycle += 1
                    time.sleep(1)
                    continue
                
                success_count = 0
                cycle_start_time = time.time()
                
                for idx, (original_formula, compiled_expr) in enumerate(compiled_batch.items(), 1):
                    logging.info(f"[Formula {idx}/{len(compiled_batch)}] Eval: {original_formula[:80]}...")
                    
                    try:
                        if isinstance(df_eager, pl.LazyFrame):
                            tmp = (
                                df_eager
                                .with_columns(alpha_signal=compiled_expr.over("ticker"))
                                .with_columns(final_alpha=pl.col("alpha_signal").over("date"))
                                .collect()
                            )
                        else:
                            tmp = (
                                df_eager
                                .with_columns(alpha_signal=compiled_expr.over("ticker"))
                                .with_columns(final_alpha=pl.col("alpha_signal").over("date"))
                            )

                        metrics = calculate_rolling_metrics(tmp, window_sizes=[252, 504], step_days=63)
                        
                        if metrics["windows"] == 0:
                            raise ValueError("No valid windows evaluated (maybe all nulls?)")
                            
                        alpha_logger.log_processed(original_formula, metrics, success=True)
                        success_count += 1
                        
                        if (metrics['median_sharpe'] >= SHARPE_THRESHOLD and 
                            metrics['worst_sharpe'] > WORST_SHARPE_MIN and
                            metrics['consistency'] >= CONSISTENCY_MIN and
                            metrics['mean_turnover'] < TURNOVER_MAX and 
                            metrics['mean_short_pct'] > SHORT_MIN):
                            
                            found_count += 1
                            alpha_logger.log_elite(original_formula, metrics)
                                
                    except Exception as e:
                        error_msg = f"Execution Error: {str(e)[:100]}"
                        alpha_logger.log_processed(original_formula, {}, success=False, error=error_msg)
                        continue
                
                cycle_duration = time.time() - cycle_start_time
                logging.info(f"[CYCLE {cycle}] Executed {success_count}/{BATCH_SIZE} formulas. Total elite: {found_count}")
                cycle += 1
                
            except KeyboardInterrupt:
                logging.info("Keyboard interrupt received. Shutting down gracefully...")
                logging.info(f"\n[FINAL STATISTICS]")
                logging.info(f"  Total Generated: {alpha_logger.generated_count}")
                logging.info(f"  Total Processed: {alpha_logger.processed_count}")
                logging.info(f"  Elite Alphas: {alpha_logger.elite_count}")
                break
            except Exception as e:
                logging.error(f"Unexpected error in cycle {cycle}: {e}", exc_info=True)
                cycle += 1
                time.sleep(5)
                continue
                
    except Exception as e:
        logging.error(f"Fatal error in main loop: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()