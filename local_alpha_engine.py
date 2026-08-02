import os
import glob
import random
import json
import polars as pl
from dataclasses import dataclass
from typing import List, Dict
import math

# --- 1. AST Data Structures ---
@dataclass
class ASTNode:
    type: str  
    value: str 
    children: List['ASTNode'] = None

    def to_string(self) -> str:
        if self.type in ('field', 'constant'):
            return str(self.value)
        elif self.type == 'operator':
            args = ", ".join(child.to_string() for child in (self.children or []))
            return f"{self.value}({args})"
        return ""

# --- 2. Polars Data Loader ---
def load_local_data(data_dir: str = "data/raw") -> pl.LazyFrame:
    print(f"Loading parquet files from {data_dir}...")
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}.")
        
    dataframes = []
    for file in parquet_files:
        ticker = os.path.basename(file).replace(".parquet", "")
        df = pl.scan_parquet(file)
        
        cols = df.collect_schema().names()
        rename_map = {}
        for col in cols:
            base_name = col.split('_')[0].lower()
            if 'date' in base_name: 
                rename_map[col] = 'date'
            elif base_name in ['open', 'high', 'low', 'close', 'volume']:
                if base_name not in rename_map.values():
                    rename_map[col] = base_name
                    
        df = df.rename(rename_map)
        core_cols = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume'] if c in df.collect_schema().names()]
        df = df.select(core_cols).with_columns(pl.lit(ticker).alias("ticker"))
        
        cast_map = {col: pl.Float64 for col in core_cols if col != 'date'}
        df = df.cast(cast_map)
        dataframes.append(df)
        
    panel_df = pl.concat(dataframes, how="diagonal")
    
    # Filter to 2017 to current date
    # This provides a longer historical period for robust alpha evaluation
    panel_df = panel_df.filter(
        pl.col("date") >= pl.date(2017, 1, 1)
    )
    
    return panel_df.sort(["date", "ticker"])

# --- 3. Custom Alpha Generator with Enforced Constraints ---
class NativeAlphaGenerator:
    def __init__(self, raw_operators_path: str, valid_fields: List[str]):
        self.valid_fields = valid_fields
        self.operators = self._load_operators(raw_operators_path)
        
        self.blacklist = ['combo_a', 'vector_neut']
        
        # Only use operators that are actually implemented in compile_to_polars
        self.implemented_ops = {
            'add', 'subtract', 'multiply', 'divide',
            'ts_delay', 'ts_delta', 'ts_mean', 'ts_decay_linear', 'ts_max', 'ts_min', 'ts_std_dev',
            'ts_covariance', 'ts_corr',
            'rank', 'normalize', 'zscore', 'abs', 'sign', 'reverse'
        }
        
        # Manually classify operators by arity (number of expression parameters)
        # Note: ts_delay, ts_delta, ts_mean, etc. take 1 expression + 1 constant (window)
        # ts_covariance, ts_corr take 2 expressions + 1 constant (window)
        self.unary_ops = ['rank', 'normalize', 'zscore', 'abs', 'sign', 'reverse']
        self.binary_ops = ['add', 'subtract', 'multiply', 'divide']
        self.unary_with_window = ['ts_delay', 'ts_delta', 'ts_mean', 'ts_decay_linear', 'ts_max', 'ts_min', 'ts_std_dev']
        self.binary_with_window = ['ts_covariance', 'ts_corr']
        
        if not self.unary_ops: self.unary_ops = ['ts_mean', 'rank', 'zscore', 'abs']
        if not self.binary_ops: self.binary_ops = ['add', 'subtract', 'divide']

    def _load_operators(self, path: str) -> List[Dict]:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return []

    def generate_ast(self) -> ASTNode:
        # Build the chaotic, randomized mathematical core
        core_ast = self._build_random_core()
        
        # IRONCLAD WRAPPERS: Force Market Neutrality & Low Turnover
        # 1. Normalize: Forces cross-sectional mean to 0 (Balances Long/Short perfectly)
        neutral_node = ASTNode(type="operator", value="normalize", children=[core_ast])
        
        # 2. TS_Decay_Linear or TS_Mean: Smooths the signal to crush turnover below 50%
        # Randomly choose between a 5, 10, or 20 day smoothing window
        lookback = str(random.choice([5, 10, 20]))
        lookback_node = ASTNode(type="constant", value=lookback)
        
        smoothing_op = random.choice(["ts_decay_linear", "ts_mean"])
        final_wrapped_ast = ASTNode(type="operator", value=smoothing_op, children=[neutral_node, lookback_node])
        
        return final_wrapped_ast

    def _build_random_core(self) -> ASTNode:
        target_operators = random.randint(2, 5)
        max_depth = 5 
        
        def build_tree(depth: int, op_count: int) -> tuple[ASTNode, int]:
            if depth >= max_depth or op_count >= target_operators:
                return ASTNode(type="field", value=random.choice(self.valid_fields)), op_count
            
            if depth > 1 and op_count >= 3 and random.random() < 0.2:
                return ASTNode(type="field", value=random.choice(self.valid_fields)), op_count
                
            left_node, op_count = build_tree(depth + 1, op_count)
            
            # Choose operator type based on what's available
            op_type = random.choice(['unary', 'binary', 'unary_window', 'binary_window'])
            
            if op_type == 'binary' and self.binary_ops and op_count < target_operators and random.random() < 0.7:
                op = random.choice(self.binary_ops)
                right_node, op_count = build_tree(depth + 1, op_count)
                node = ASTNode(type="operator", value=op, children=[left_node, right_node])
                op_count += 1
            elif op_type == 'binary_window' and self.binary_with_window and op_count < target_operators and random.random() < 0.7:
                op = random.choice(self.binary_with_window)
                right_node, op_count = build_tree(depth + 1, op_count)
                lookback_node = ASTNode(type="constant", value=str(random.choice([5, 10, 20, 60])))
                node = ASTNode(type="operator", value=op, children=[left_node, right_node, lookback_node])
                op_count += 1
            elif op_type == 'unary_window' and self.unary_with_window:
                op = random.choice(self.unary_with_window)
                lookback_node = ASTNode(type="constant", value=str(random.choice([5, 10, 20, 60])))
                node = ASTNode(type="operator", value=op, children=[left_node, lookback_node])
                op_count += 1
            else:
                op = random.choice(self.unary_ops)
                node = ASTNode(type="operator", value=op, children=[left_node])
                op_count += 1
                
            return node, op_count
            
        core, _ = build_tree(0, 0)
        return core

# --- 4. The Translator (AST -> Polars Expr) ---
def compile_to_polars(node: ASTNode) -> pl.Expr:
    if node.type == 'constant':
        return pl.lit(int(node.value) if node.value.isdigit() else float(node.value))
    elif node.type == 'field':
        return pl.col(node.value)
    
    elif node.type == 'operator':
        op = node.value
        args = [compile_to_polars(c) for c in node.children]

        if op == 'add': return args[0] + args[1]
        if op == 'subtract': return args[0] - args[1]
        if op == 'multiply': return args[0] * args[1]
        if op == 'divide': return args[0] / args[1]

        if op == 'ts_delay': return args[0].shift(int(node.children[1].value))
        if op == 'ts_delta': return args[0] - args[0].shift(int(node.children[1].value))
        if op == 'ts_mean': return args[0].rolling_mean(window_size=int(node.children[1].value))
        if op == 'ts_decay_linear': 
            # Local proxy: Simple rolling mean approximates the smoothing effect for local turnover checks
            return args[0].rolling_mean(window_size=int(node.children[1].value))
        if op == 'ts_max': return args[0].rolling_max(window_size=int(node.children[1].value))
        if op == 'ts_min': return args[0].rolling_min(window_size=int(node.children[1].value))
        if op == 'ts_std_dev': return args[0].rolling_std(window_size=int(node.children[1].value))

        if op == 'ts_covariance': return pl.rolling_cov(args[0], args[1], window_size=int(node.children[2].value))
        if op == 'ts_corr': return pl.rolling_corr(args[0], args[1], window_size=int(node.children[2].value))

        if op == 'rank': return args[0].rank() / args[0].count()
        if op == 'normalize': return args[0] - args[0].mean()
        if op == 'zscore': return (args[0] - args[0].mean()) / args[0].std()
        if op == 'abs': return args[0].abs()
        if op == 'sign': return args[0].sign()
        if op == 'reverse': return -args[0]

        raise NotImplementedError(f"Operator '{op}' is not implemented in Polars compiler.")

# --- 5. Brain Simulator Metrics ---
def calculate_brain_metrics(df: pl.DataFrame) -> dict:
    if len(df) == 0: 
        print("Warning: Empty dataframe after evaluation")
        return {"sharpe": 0.0, "turnover": 0.0, "short_pct": 0.0}

    # Check for NaN values in alpha signal
    nan_count = df["final_alpha"].is_null().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values in final_alpha out of {len(df)} rows")

    # 1. Forward Returns (Delay 1: trade at T+1 close, return from T+1 close to T+2 close)
    df_temp = df.with_columns(
        pl.col("close").shift(-1).over("ticker").alias("trade_close"),
        pl.col("close").shift(-2).over("ticker").alias("exit_close")
    )
    # Keep only rows where both trade and exit prices exist (avoid broad drop_nulls)
    df_eval = df_temp.filter(
        pl.col("trade_close").is_not_null() & pl.col("exit_close").is_not_null()
    ).with_columns(
        (((pl.col("exit_close") - pl.col("trade_close")) / pl.col("trade_close")).alias("fwd_return"))
    )

    if df_eval.height == 0:
        print("Warning: Empty dataframe after forward returns calculation")
        return {"sharpe": 0.0, "turnover": 0.0, "short_pct": 0.0}

    # 2. Booksize Normalization (Sum of absolute weights = 1.0 per day)
    try:
        # Compute per-date denom safely and avoid division by zero
        df_eval = df_eval.with_columns(
            pl.col("final_alpha").abs().sum().over("date").alias("abs_sum")
        ).with_columns(
            pl.when(pl.col("abs_sum") == 0.0)
              .then(0.0)
              .otherwise(pl.col("final_alpha") / pl.col("abs_sum"))
              .alias("weight")
        )
        # Keep rows where weight and forward return are present
        df_eval = df_eval.filter(pl.col("weight").is_not_null() & pl.col("fwd_return").is_not_null())
    except Exception as e:
        print(f"Warning: Error in weight normalization: {e}")
        return {"sharpe": 0.0, "turnover": 0.0, "short_pct": 0.0}

    if df_eval.height == 0:
        print("Warning: Empty dataframe after weight normalization")
        return {"sharpe": 0.0, "turnover": 0.0, "short_pct": 0.0}

    # 3. Calculate Portfolio Returns and Position Counts
    daily_stats = df_eval.group_by("date").agg(
        (pl.col("weight") * pl.col("fwd_return")).sum().alias("port_return"),
        (pl.col("weight") > 0).sum().alias("long_count"),
        (pl.col("weight") < 0).sum().alias("short_count")
    ).sort("date")

    # 4. Calculate Turnover (Daily absolute change in weights)
    turnover_df = df_eval.sort(["ticker", "date"]).with_columns(
        pl.col("weight").shift(1).over("ticker").alias("prev_weight")
    ).fill_null(0.0)

    daily_turnover = turnover_df.group_by("date").agg(
        (pl.col("weight") - pl.col("prev_weight")).abs().sum().alias("daily_turnover")
    )

    # Aggregate Final Metrics
    mean_ret = daily_stats["port_return"].mean()
    std_ret = daily_stats["port_return"].std()
    
    # Handle invalid statistics explicitly rather than using truthiness
    if mean_ret is None or std_ret is None or std_ret == 0.0 or math.isnan(mean_ret) or math.isnan(std_ret) or math.isinf(mean_ret) or math.isinf(std_ret):
        print(f"Warning: Invalid return statistics - mean_ret={mean_ret}, std_ret={std_ret}")
        sharpe = 0.0
    else:
        sharpe = (mean_ret / std_ret) * math.sqrt(252)

    # Divide by 2 because shifting 100% of a portfolio equals 2.0 total weight change
    avg_turnover = 0.0
    if daily_turnover.height > 0:
        avg_turnover = (float(daily_turnover["daily_turnover"].mean()) / 2.0) * 100.0
    
    avg_longs = float(daily_stats["long_count"].mean()) if daily_stats.height > 0 else 0.0
    avg_shorts = float(daily_stats["short_count"].mean()) if daily_stats.height > 0 else 0.0
    total_positions = avg_longs + avg_shorts
    short_pct = (avg_shorts / total_positions) * 100 if total_positions and total_positions > 0 else 0.0

    return {
        "sharpe": sharpe,
        "turnover": avg_turnover,
        "short_pct": short_pct
    }

# --- 6. Execution Pipeline ---
if __name__ == "__main__":
    df_lazy = load_local_data()
    # Collect once to avoid repeated parquet scans during quick native test
    try:
        df_eager = df_lazy.collect()
    except Exception:
        df_eager = df_lazy

    local_fields = ["open", "high", "low", "close", "volume"]
    generator = NativeAlphaGenerator(raw_operators_path="operatorRAW.json", valid_fields=local_fields)
    
    print("\n--- Generating and Evaluating Brain-Compliant Alpha ---")
    
    alpha_ast = generator.generate_ast()
    formula = alpha_ast.to_string()
    print(f"Generated Formula: \n{formula}\n")
    
    polars_expr = compile_to_polars(alpha_ast)
    
    # Apply expression and compute final_alpha; avoid dropping rows en masse
    if isinstance(df_eager, pl.LazyFrame):
        tmp_df = (
            df_eager
            .with_columns(alpha_signal=polars_expr.over("ticker"))
            .with_columns(final_alpha=pl.col("alpha_signal").over("date"))
            .collect()
        )
    else:
        tmp_df = (
            df_eager
            .with_columns(alpha_signal=polars_expr.over("ticker"))
            .with_columns(final_alpha=pl.col("alpha_signal").over("date"))
        )

    evaluated_df = tmp_df.filter(pl.col("alpha_signal").is_not_null() & pl.col("final_alpha").is_not_null())
    
    metrics = calculate_brain_metrics(evaluated_df)
    
    print("="*40)
    print(f"SHARPE RATIO : {metrics['sharpe']:.4f}")
    print(f"TURNOVER     : {metrics['turnover']:.2f}%  (Target: < 50%)")
    print(f"SHORT EXP    : {metrics['short_pct']:.2f}%  (Target: > 40%)")
    print("="*40)