import time
import csv
import os
import logging
import ast
import math
from datetime import datetime
import polars as pl
from dotenv import load_dotenv

load_dotenv() 

# Import your data loader and metrics from the engine
from local_alpha_engine import AlphaGenerator, load_local_data, calculate_rolling_metrics, compile_to_polars, split_data_temporal, materialize_subtrees, evaluate_ast
from operator_registry import OperatorRegistry
from alpha_validator import validate_ast
from alpha_templates import AlphaTemplateGenerator
from alpha_healer import AlphaHealer
from alpha_evolver import AlphaEvolver
from ast_node import ASTNode, parse_formula
from alpha_cache import AlphaMetricsCache, SubTreeColumnCache
from benchmark_observer import BenchmarkObserver
import argparse
import gc

# --- Configuration ---
SHARPE_THRESHOLD = 1.0
WORST_SHARPE_MIN = 0.0
CONSISTENCY_MIN = 60.0
TURNOVER_MAX = 50.0
SHORT_MIN = 40.0
SHORT_MAX = 60.0
BATCH_SIZE = 5

# --- Out-of-Sample Validation Thresholds ---
# Intentionally relaxed vs IS: the point is to confirm the alpha
# has *any* predictive signal on unseen data, not to re-apply the same bar.
OOS_SHARPE_THRESHOLD = 0.5
OOS_WORST_SHARPE_MIN = -0.5
OOS_CONSISTENCY_MIN = 40.0
TRAIN_PCT = 0.70  # 70% IS, 30% OOS by date

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
    
    def log_elite(self, formula, metrics, oos_metrics=None):
        self.elite_count += 1
        file_exists = os.path.isfile(ELITE_LOG_FILE)
        with open(ELITE_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "elite_id", "formula", 
                    "is_sharpe", "is_worst_sharpe", "is_consistency", "is_turnover", "is_short_pct", "is_windows",
                    "oos_sharpe", "oos_worst_sharpe", "oos_consistency", "oos_windows"
                ])
            oos_row = [
                round(oos_metrics['median_sharpe'], 4) if oos_metrics else "",
                round(oos_metrics['worst_sharpe'], 4) if oos_metrics else "",
                round(oos_metrics['consistency'], 2) if oos_metrics else "",
                oos_metrics['windows'] if oos_metrics else ""
            ]
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
            ] + oos_row)
        
        file_exists = os.path.isfile(BEST_ALPHAS_FILE)
        with open(BEST_ALPHAS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "is_sharpe", "is_worst_sharpe", "is_consistency", 
                    "turnover", "short_pct",
                    "oos_sharpe", "oos_worst_sharpe", "oos_consistency",
                    "formula"
                ])
            oos_vals = [
                round(oos_metrics['median_sharpe'], 4) if oos_metrics else "",
                round(oos_metrics['worst_sharpe'], 4) if oos_metrics else "",
                round(oos_metrics['consistency'], 2) if oos_metrics else "",
            ]
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                round(metrics['median_sharpe'], 4),
                round(metrics['worst_sharpe'], 4),
                round(metrics['consistency'], 2),
                round(metrics['mean_turnover'], 2),
                round(metrics['mean_short_pct'], 2),
            ] + oos_vals + [formula])
        
        oos_info = ""
        if oos_metrics:
            oos_info = f" | OOS={oos_metrics['median_sharpe']:.2f}"
        logging.info(f"[ELITE #{self.elite_count}] IS={metrics['median_sharpe']:.2f} | Turn={metrics['mean_turnover']:.1f}%{oos_info}")

alpha_logger = AlphaLogger()


class MultipleTestingTracker:
    """
    Tracks the number of formulas tested and computes adjusted significance
    thresholds to account for multiple-testing / data-mining bias.
    
    Problem: Testing 10,000 random formulas on the same historical data will
    produce some with Sharpe > 1.0 purely by chance. The more formulas tested,
    the higher the bar must be to have confidence the signal is real.
    
    Uses a practical Bonferroni-inspired adjustment:
        adjusted_sharpe = base_threshold + 0.3 * log10(N_tested)
    
    After 100 tests:   threshold ~ 1.6
    After 1,000 tests: threshold ~ 1.9
    After 10,000 tests: threshold ~ 2.2
    
    Reference: Harvey, Liu, Zhu (2016) "...and the Cross-Section of Expected Returns"
    """
    def __init__(self, base_sharpe: float = SHARPE_THRESHOLD):
        self.n_tested = 0
        self.base_sharpe = base_sharpe
        self.n_is_passed = 0
        self.n_oos_passed = 0
        
    def record_test(self, count: int = 1):
        self.n_tested += count
        
    def record_is_pass(self):
        self.n_is_passed += 1
        
    def record_oos_pass(self):
        self.n_oos_passed += 1
        
    def get_adjusted_threshold(self) -> float:
        """Returns the Sharpe threshold adjusted for number of tests run."""
        if self.n_tested <= 1:
            return self.base_sharpe
        return self.base_sharpe + 0.3 * math.log10(self.n_tested)
    
    def get_report(self) -> str:
        adj = self.get_adjusted_threshold()
        oos_rate = (self.n_oos_passed / self.n_is_passed * 100) if self.n_is_passed > 0 else 0
        return (
            f"  Formulas tested:       {self.n_tested}\n"
            f"  Base Sharpe threshold: {self.base_sharpe:.2f}\n"
            f"  Adjusted threshold:    {adj:.2f} (inflation: {adj/self.base_sharpe:.1f}x)\n"
            f"  IS passes:             {self.n_is_passed}\n"
            f"  OOS passes:            {self.n_oos_passed}\n"
            f"  OOS survival rate:     {oos_rate:.1f}%"
        )

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
    
    print("\nIN-SAMPLE ELITE THRESHOLDS:")
    print(f"  Median Sharpe >= {SHARPE_THRESHOLD} (auto-adjusts for multiple testing)")
    print(f"  Worst Sharpe  >  {WORST_SHARPE_MIN}")
    print(f"  Consistency   >= {CONSISTENCY_MIN}%")
    print(f"  Turnover      <  {TURNOVER_MAX}%")
    print(f"  Short %       >  {SHORT_MIN}%")
    
    print("\nOUT-OF-SAMPLE VALIDATION THRESHOLDS:")
    print(f"  OOS Median Sharpe >= {OOS_SHARPE_THRESHOLD}")
    print(f"  OOS Worst Sharpe  >  {OOS_WORST_SHARPE_MIN}")
    print(f"  OOS Consistency   >= {OOS_CONSISTENCY_MIN}%")
    print(f"  Train/Test Split:    {TRAIN_PCT*100:.0f}% / {(1-TRAIN_PCT)*100:.0f}%")
    print("=" * 58 + "\n")

def generate_candidate_batch(
    strategy: str,
    template_gen: AlphaTemplateGenerator,
    evolver: AlphaEvolver,
    healer: AlphaHealer,
    random_gen: AlphaGenerator,
    registry: OperatorRegistry,
    valid_fields: list,
    batch_size: int = 5
) -> list:
    """
    Generates a batch of (ast_node, formula_str, source_tag) tuples according to the selected strategy.
    """
    candidates = []

    def _try_add(node, tag):
        if node and len(candidates) < batch_size:
            # Enforce outer cross-sectional neutralization if outer operator is time-series or drift-inducing
            # Guarantees dollar-neutrality parity with WorldQuant Brain
            if node.type == 'operator' and (node.value.startswith('ts_') or node.value in ('reverse', 'sign', 'abs')):
                node = ASTNode(type='operator', value='zscore', children=[node])
            is_valid, _ = validate_ast(node, registry, valid_fields)
            if is_valid:
                candidates.append((node, node.to_string(), tag))

    if strategy == "templates":
        for _ in range(batch_size * 2):
            if len(candidates) >= batch_size: break
            _try_add(template_gen.generate(apply_wrappers=True), "Template")

    elif strategy == "evolve":
        for node in evolver.generate_batch(count=batch_size):
            _try_add(node, "Evolver")

    elif strategy == "heal":
        for node in healer.generate_healed_batch(batch_size=batch_size):
            _try_add(node, "Healer")
        while len(candidates) < batch_size:
            _try_add(template_gen.generate(apply_wrappers=True), "Template(Backfill)")

    elif strategy == "random":
        attempts = 0
        while len(candidates) < batch_size and attempts < batch_size * 5:
            attempts += 1
            _try_add(random_gen.generate_ast(), "Random")

    else:
        # Default: "hybrid" (40% templates, 40% evolution, 20% healing)
        # 1. Healer (1 candidate)
        for h in healer.generate_healed_batch(batch_size=1):
            _try_add(h, "Healer")

        # 2. Evolver (2 candidates)
        for ev in evolver.generate_batch(count=2):
            _try_add(ev, "Evolver")

        # 3. Templates (fill remaining slots up to batch_size)
        while len(candidates) < batch_size:
            _try_add(template_gen.generate(apply_wrappers=True), "Template")

    # Safety backfill if still under batch_size
    attempts = 0
    while len(candidates) < batch_size and attempts < 10:
        attempts += 1
        _try_add(random_gen.generate_ast(), "Random(Fallback)")

    return candidates


def main(strategy: str = "hybrid", max_cycles: int = 0):
    alpha_logger = AlphaLogger()
    
    try:
        # Load registry
        registry = OperatorRegistry("operatorRAW.json")
        
        # Initialize Data
        df_lazy = load_local_data()
        try:
            df_eager = df_lazy.collect()
            logging.info(f"Data loaded into memory. Total rows: {df_eager.height}")
        except Exception as e:
            logging.warning(f"Could not collect lazy frame into memory: {e}. Collecting now.")
            df_eager = df_lazy.collect()
            
        # --- Temporal Split: In-Sample / Out-of-Sample ---
        is_df, oos_df, split_date = split_data_temporal(df_eager, train_pct=TRAIN_PCT)
        base_is_df = is_df.clone()
        logging.info(f"Temporal split at {split_date}")
        logging.info(f"  In-Sample:      {is_df.height:,} rows ({TRAIN_PCT*100:.0f}%)")
        logging.info(f"  Out-of-Sample:  {oos_df.height:,} rows ({(1-TRAIN_PCT)*100:.0f}%)")
        
        local_fields = ["open", "high", "low", "close", "volume", "vwap"]
        random_gen = AlphaGenerator(registry, valid_fields=local_fields)
        template_gen = AlphaTemplateGenerator(valid_fields=local_fields)
        evolver = AlphaEvolver(registry, valid_fields=local_fields, population_size=40)
        healer = AlphaHealer(registry, logs_csv_path=PROCESSED_LOG_FILE, valid_fields=local_fields)

        evolver_seeded = evolver.seed_from_logs(PROCESSED_LOG_FILE, top_n=20)
        evolver.seed_from_templates(count=15)
        logging.info(f"Initialized Strategy Engine [{strategy.upper()}]:")
        logging.info(f"  Templates: 6 Market Archetypes (Mean Rev, Mom, Vol, Vol-Price, Dips, Combos)")
        logging.info(f"  Evolver:   Population={len(evolver.population)} (Seeded from logs={evolver_seeded})")
        logging.info(f"  Healer:    {healer.get_candidate_count()} Near-miss candidates in pool")

        # Multiple-testing tracker — adjusts thresholds as more formulas are tested
        mt_tracker = MultipleTestingTracker(base_sharpe=SHARPE_THRESHOLD)
        
        # 2-Tier Caching Infrastructure
        # Tier 1: Top-level formula string -> evaluated metrics
        metrics_cache = AlphaMetricsCache(max_size=100_000)
        preloaded_count = metrics_cache.preload_from_csv(PROCESSED_LOG_FILE)
        logging.info(f"  Tier-1 Metrics Cache: {preloaded_count} formulas preloaded from {PROCESSED_LOG_FILE}")
        
        # Tier 2: Recursive sub-tree column cache with 30-column cap and 2GB RAM max cap
        RAM_CAP_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
        subtree_cache = SubTreeColumnCache(max_ram_bytes=RAM_CAP_BYTES, max_columns=30)
        logging.info(f"  Tier-2 SubTree Cache: Enabled with max 30 columns and {RAM_CAP_BYTES / (1024**3):.1f}GB RAM Cap")

        # Performance & Benchmark Observer
        observer = BenchmarkObserver()

        run_diagnostics(registry, df_lazy)
        
    except Exception as e:
        logging.error(f"Initialization failed: {e}", exc_info=True)
        return

    logging.info(f"Starting Alpha Discovery Loop (Strategy: {strategy.upper()})...")
    found_count = 0
    cycle = 1
    
    try:
        while True:
            if max_cycles > 0 and cycle > max_cycles:
                logging.info(f"Reached max cycles ({max_cycles}). Stopping gracefully.")
                break

            logging.info(f"--- Cycle {cycle} [{strategy.upper()}] ---")
            
            try:
                # Generate candidate batch using selected strategy
                batch_items = generate_candidate_batch(
                    strategy=strategy,
                    template_gen=template_gen,
                    evolver=evolver,
                    healer=healer,
                    random_gen=random_gen,
                    registry=registry,
                    valid_fields=local_fields,
                    batch_size=BATCH_SIZE
                )

                if not batch_items:
                    logging.warning("Could not generate valid ASTs. Check operator registry.")
                    time.sleep(1)
                    continue

                for ast_node, formula_str, source_tag in batch_items:
                    alpha_logger.log_generated(formula_str)

                success_count = 0
                cycle_start_time = time.time()
                
                for idx, (ast_node, original_formula, source_tag) in enumerate(batch_items, 1):
                    alpha_eval_start = time.time()

                    # --- Tier 1 Cache: Formula String -> Metrics ---
                    cached_metrics = metrics_cache.get(original_formula)
                    if cached_metrics:
                        logging.info(f"[Formula {idx}/{len(batch_items)}] [{source_tag}] [TIER-1 CACHE HIT] {original_formula[:75]}...")
                        is_metrics = cached_metrics
                        success_count += 1
                        evolver.add_individual(ast_node, is_metrics)
                        observer.record_evaluation(time.time() - alpha_eval_start, hit_tier1=True)
                        continue

                    logging.info(f"[Formula {idx}/{len(batch_items)}] [{source_tag}] Eval: {original_formula[:75]}...")
                    
                    try:
                        # --- Stage 1: In-Sample Discovery ---
                        # Evaluate strictly partitioned: time-series over('ticker'), cross-sectional over('date')
                        t_eval0 = time.time()
                        is_tmp, final_expr = evaluate_ast(is_df, ast_node, subtree_cache=subtree_cache)
                        is_tmp = is_tmp.with_columns(final_alpha=final_expr)
                        observer.record_stage("polars_eval_ms", time.time() - t_eval0)

                        t_met0 = time.time()
                        is_metrics = calculate_rolling_metrics(is_tmp, window_sizes=[252, 504], step_days=63)
                        observer.record_stage("rolling_metrics_ms", time.time() - t_met0)
                        mt_tracker.record_test()
                        
                        if is_metrics["windows"] == 0:
                            raise ValueError("No valid windows evaluated (maybe all nulls?)")
                            
                        # Save in Tier 1 cache
                        metrics_cache.set(original_formula, is_metrics)
                        alpha_logger.log_processed(original_formula, is_metrics, success=True)
                        success_count += 1
                        observer.record_evaluation(time.time() - alpha_eval_start)
                        
                        # Continuous learning loop: Feed newly evaluated metrics to Evolver and Healer
                        evolver.add_individual(ast_node, is_metrics)
                        if (0.45 <= is_metrics['median_sharpe'] < 1.0) or (is_metrics['median_sharpe'] <= -0.5) or (is_metrics['consistency'] >= 60.0 and is_metrics['median_sharpe'] >= 0.3):
                            healer.near_misses.append({
                                'formula': original_formula,
                                'median_sharpe': is_metrics['median_sharpe'],
                                'worst_sharpe': is_metrics['worst_sharpe'],
                                'consistency': is_metrics['consistency']
                            })

                        # Adjusted threshold rises as more formulas are tested.
                        # This counteracts the multiple-testing problem.
                        adjusted_sharpe = mt_tracker.get_adjusted_threshold()
                        
                        if (is_metrics['median_sharpe'] >= adjusted_sharpe and 
                            is_metrics['worst_sharpe'] > WORST_SHARPE_MIN and
                            is_metrics['consistency'] >= CONSISTENCY_MIN and
                            is_metrics['mean_turnover'] < TURNOVER_MAX and 
                            SHORT_MIN <= is_metrics['mean_short_pct'] <= SHORT_MAX):
                            
                            mt_tracker.record_is_pass()
                            logging.info(
                                f"  [IS PASS] [{source_tag}] Sharpe={is_metrics['median_sharpe']:.2f} "
                                f"(adj_thresh={adjusted_sharpe:.2f}). Validating on OOS..."
                            )
                            
                            # --- Stage 2: Out-of-Sample Validation ---
                            # Independent test on unseen data without lookahead bias
                            oos_tmp, oos_expr = evaluate_ast(oos_df, ast_node)
                            oos_tmp = oos_tmp.with_columns(final_alpha=oos_expr)
                            oos_metrics = calculate_rolling_metrics(oos_tmp, window_sizes=[252], step_days=63)
                            
                            if (oos_metrics['median_sharpe'] >= OOS_SHARPE_THRESHOLD and
                                oos_metrics['worst_sharpe'] > OOS_WORST_SHARPE_MIN and
                                oos_metrics['consistency'] >= OOS_CONSISTENCY_MIN):
                                
                                mt_tracker.record_oos_pass()
                                found_count += 1
                                alpha_logger.log_elite(original_formula, is_metrics, oos_metrics=oos_metrics)
                                logging.info(
                                    f"  [OOS PASS] [{source_tag}] OOS_Sharpe={oos_metrics['median_sharpe']:.2f} "
                                    f"| Genuinely elite (survives out-of-sample)"
                                )
                            else:
                                logging.info(
                                    f"  [OOS FAIL] [{source_tag}] OOS_Sharpe={oos_metrics['median_sharpe']:.2f} "
                                    f"— likely overfit to in-sample data"
                                )
                                
                    except Exception as e:
                        error_msg = f"Execution Error: {str(e)[:100]}"
                        alpha_logger.log_processed(original_formula, {}, success=False, error=error_msg)
                        continue
                
                cycle_duration = time.time() - cycle_start_time
                adj_thresh = mt_tracker.get_adjusted_threshold()
                c1_stats = metrics_cache.stats()
                c2_stats = subtree_cache.stats()
                observer.flush_to_disk()
                logging.info(
                    f"[CYCLE {cycle}] Eval {success_count}/{len(batch_items)} | "
                    f"Elite: {found_count} | Adj.Sharpe: {adj_thresh:.2f} | "
                    f"Cache T1 Hits: {c1_stats['hits']} ({c1_stats['hit_rate_pct']}%) | "
                    f"T2 Subtrees: {c2_stats['cached_subtrees']} (Hits: {c2_stats['hits']}) | "
                    f"Time: {cycle_duration:.2f}s"
                )

                # End of cycle memory management: release unreferenced Polars Series
                gc.collect()

                # Periodic baseline refresh every 20 cycles: drops temporary columns and reclaims Arrow memory
                if cycle % 20 == 0:
                    is_df = base_is_df.clone()
                    subtree_cache.clear()
                    gc.collect()
                    logging.info(f"[CYCLE {cycle}] Process memory reclaimed and baseline DataFrame refreshed.")

                cycle += 1
                
            except KeyboardInterrupt:
                logging.info("Keyboard interrupt received. Shutting down gracefully...")
                observer.flush_to_disk()
                logging.info(f"\n[FINAL STATISTICS]")
                logging.info(f"  Total Generated:  {alpha_logger.generated_count}")
                logging.info(f"  Total Processed:  {alpha_logger.processed_count}")
                logging.info(f"  Elite (OOS-validated): {alpha_logger.elite_count}")
                logging.info(f"  Tier 1 Cache: {metrics_cache.stats()}")
                logging.info(f"  Tier 2 Cache: {subtree_cache.stats()}")
                logging.info(f"\n[MULTIPLE TESTING REPORT]")
                logging.info(f"\n{mt_tracker.get_report()}")
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
    parser = argparse.ArgumentParser(description="BrainQuant Alpha Background Runner")
    parser.add_argument(
        "--strategy", "-s",
        choices=["hybrid", "templates", "evolve", "heal", "random"],
        default="hybrid",
        help="Alpha generation strategy (default: hybrid)"
    )
    parser.add_argument(
        "--max-cycles", "-c",
        type=int,
        default=0,
        help="Maximum cycles to run (0 = infinite continuous search)"
    )
    args = parser.parse_args()
    main(strategy=args.strategy, max_cycles=args.max_cycles)