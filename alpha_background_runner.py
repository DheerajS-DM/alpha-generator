import time
import csv
import os
import logging
import ast
from datetime import datetime
import polars as pl

print("1. Imports successful...")

# --- NEW: Load Environment Variables ---
from dotenv import load_dotenv
load_dotenv() 
print("2. Dotenv loaded...")

# Import your data loader and metrics from the engine
from local_alpha_engine import NativeAlphaGenerator, load_local_data, calculate_brain_metrics, compile_to_polars
print("3. Local modules imported successfully...")

# --- Configuration ---
SHARPE_THRESHOLD = 1.0
TURNOVER_MAX = 50.0
SHORT_MIN = 40.0
BATCH_SIZE = 3  # Further reduced to minimize rate limiting

BEST_ALPHAS_FILE = "elite_alphas.csv"
GENERATED_LOG_FILE = "logs_generated.csv"
PROCESSED_LOG_FILE = "logs_processed.csv"
ELITE_LOG_FILE = "logs_elite.csv"
ERROR_LOG_FILE = "transpiler_errors.log"

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ERROR_LOG_FILE),
        logging.StreamHandler()
    ]
)

# --- Comprehensive Alpha Logging System ---
class AlphaLogger:
    """Logs alphas at each stage: generated, processed, elite"""
    
    def __init__(self):
        self.generated_count = 0
        self.processed_count = 0
        self.elite_count = 0
    
    def log_generated(self, formula, translator_used):
        """Log when an alpha is generated"""
        self.generated_count += 1
        file_exists = os.path.isfile(GENERATED_LOG_FILE)
        with open(GENERATED_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "generation_id", "formula", 
                    "formula_length", "translator"
                ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                self.generated_count,
                formula,
                len(formula),
                translator_used
            ])
        logging.info(f"[GEN #{self.generated_count}] Formula generated")
    
    def log_processed(self, formula, polars_code, metrics, translator_used, success=True, error=None):
        """Log when an alpha is successfully processed/executed"""
        if success:
            self.processed_count += 1
            file_exists = os.path.isfile(PROCESSED_LOG_FILE)
            with open(PROCESSED_LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "timestamp", "processed_id", "formula", 
                        "polars_code", "sharpe", "turnover", "short_pct",
                        "translator"
                    ])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    self.processed_count,
                    formula,
                    polars_code,
                    round(metrics['sharpe'], 4),
                    round(metrics['turnover'], 2),
                    round(metrics['short_pct'], 2),
                    translator_used
                ])
            logging.info(f"[OK] [PROC #{self.processed_count}] Sharpe={metrics['sharpe']:.2f} | Turn={metrics['turnover']:.1f}% | Short={metrics['short_pct']:.1f}%")
        else:
            logging.warning(f"[FAIL] [PROC] {error}")
    
    def log_elite(self, formula, polars_code, metrics, translator_used):
        """Log when an elite alpha is found"""
        self.elite_count += 1
        
        # Log to elite-specific file
        file_exists = os.path.isfile(ELITE_LOG_FILE)
        with open(ELITE_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "elite_id", "formula", 
                    "polars_code", "sharpe", "turnover", "short_pct",
                    "translator"
                ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                self.elite_count,
                formula,
                polars_code,
                round(metrics['sharpe'], 4),
                round(metrics['turnover'], 2),
                round(metrics['short_pct'], 2),
                translator_used
            ])
        
        # Also log to the best alphas file (for backward compat)
        file_exists = os.path.isfile(BEST_ALPHAS_FILE)
        with open(BEST_ALPHAS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "sharpe", "turnover", "short_pct", "formula"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                round(metrics['sharpe'], 4), 
                round(metrics['turnover'], 2),
                round(metrics['short_pct'], 2),
                formula
            ])
        
        logging.info(f"[ELITE #{self.elite_count}] Sharpe={metrics['sharpe']:.2f} | Turn={metrics['turnover']:.1f}% | Short={metrics['short_pct']:.1f}%")

alpha_logger = AlphaLogger()

def main():
    print("4. Inside main() function...")
    
    try:
        print("5. Initializing engine (using native compiler only)...")

        # 1. Initialize Data and Generator
        df_lazy = load_local_data()
        # Collect once to avoid repeated parquet scans (speeds up loop). Falls back to lazy if memory constrained.
        try:
            df_eager = df_lazy.collect()
            logging.info("Data loaded into memory (eager).")
        except Exception as e:
            logging.warning(f"Could not collect lazy frame into memory: {e}. Using lazy evaluation.")
            df_eager = df_lazy
        local_fields = ["open", "high", "low", "close", "volume"]

        generator = NativeAlphaGenerator(
            raw_operators_path="operatorRAW.json", 
            valid_fields=local_fields
        )
        
        print("6. Engine initialized successfully!")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during initialization: {e}")
        logging.error(f"Initialization failed: {e}", exc_info=True)
        return

    logging.info("Starting Batch Native Compilation Loop...")
    logging.info(f"Targets: Sharpe >= {SHARPE_THRESHOLD} | Turnover < {TURNOVER_MAX}% | Short Exp > {SHORT_MIN}%")
    logging.info(f"Logging to: {GENERATED_LOG_FILE}, {PROCESSED_LOG_FILE}, {ELITE_LOG_FILE}")
    
    found_count = 0
    cycle = 1
    
    try:
        while True: 
            logging.info(f"--- Cycle {cycle} ---")
            
            try:
                # Generate batch of ASTs and formula strings
                batch_asts = []
                batch_formulas = []
                for _ in range(BATCH_SIZE):
                    ast_node = generator.generate_ast()
                    batch_asts.append(ast_node)
                    formula_str = ast_node.to_string()
                    batch_formulas.append(formula_str)
                    # Log as generated
                    alpha_logger.log_generated(formula_str, "native_generator")
                    
                # Compile all formulas using native compiler (skip rule-based translator - it generates bad code)
                compiled_batch = {}
                for i, ast_node in enumerate(batch_asts):
                    try:
                        polars_expr = compile_to_polars(ast_node)
                        # Store the Expr object directly, NOT as a string
                        compiled_batch[batch_formulas[i]] = polars_expr
                        logging.debug(f"  [OK] Compiled: {batch_formulas[i][:60]}...")
                    except NotImplementedError as e:
                        logging.warning(f"  [SKIP] Unsupported operator in {batch_formulas[i][:60]}... - {e}")
                    except Exception as e:
                        logging.warning(f"  [FAIL] Native compiler failed for {batch_formulas[i][:60]}... - {e}")
                
                if not compiled_batch:
                    logging.warning("Native compiler failed on all formulas. Skipping cycle.")
                    cycle += 1
                    time.sleep(1)
                    continue
                
                logging.info(f"Native compiler succeeded for {len(compiled_batch)}/{BATCH_SIZE} formulas")
                
                success_count = 0
                cycle_start_time = time.time()
                translator_used = "native_compiler"
                
                for idx, (original_formula, compiled_expr) in enumerate(compiled_batch.items(), 1):
                    logging.info(f"[Formula {idx}/{len(compiled_batch)}] Processing: {original_formula[:60]}...")
                    logging.info(f"  EXPR OBJECT: {type(compiled_expr)}")
                    
                    try:
                        # compiled_expr is already a pl.Expr object, no need to eval
                        logging.info(f"  Applying over('ticker')...")
                        # Apply expression, compute final_alpha, but don't drop all nulls indiscriminately
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

                        # Keep only rows where we actually have a signal
                        evaluated_df = tmp.filter(pl.col("alpha_signal").is_not_null() & pl.col("final_alpha").is_not_null())
                        
                        metrics = calculate_brain_metrics(evaluated_df)
                        alpha_logger.log_processed(original_formula, str(compiled_expr), metrics, translator_used, success=True)
                        success_count += 1
                        logging.info(f"  [OK] Metrics: Sharpe={metrics['sharpe']:.2f}")
                        
                        if (metrics['sharpe'] >= SHARPE_THRESHOLD and 
                            metrics['turnover'] < TURNOVER_MAX and 
                            metrics['short_pct'] > SHORT_MIN):
                            
                            found_count += 1
                            alpha_logger.log_elite(original_formula, str(compiled_expr), metrics, translator_used)
                                
                    except Exception as e:
                        error_msg = f"Execution Error: {str(e)[:100]}"
                        logging.error(f"  EXECUTION FAILED")
                        logging.error(f"  ERROR: {str(e)}")
                        alpha_logger.log_processed(original_formula, str(compiled_expr), {}, translator_used, success=False, error=error_msg)
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
                logging.info(f"\n[LOG FILES]")
                logging.info(f"  {GENERATED_LOG_FILE}")
                logging.info(f"  {PROCESSED_LOG_FILE}")
                logging.info(f"  {ELITE_LOG_FILE}")
                break
            except Exception as e:
                logging.error(f"Unexpected error in cycle {cycle}: {e}", exc_info=True)
                cycle += 1
                time.sleep(10)
                continue
                
    except Exception as e:
        logging.error(f"Fatal error in main loop: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()