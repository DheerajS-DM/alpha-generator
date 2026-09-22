import os
import glob
import random
import json
import polars as pl
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math
import datetime

from operator_registry import OperatorRegistry, OperatorMeta
from ast_node import ASTNode
from alpha_validator import validate_ast

# --- 2. Polars Data Loader (Expanded Fields) ---
def load_local_data(data_dir: str = "data/raw") -> pl.LazyFrame:
    print(f"Loading parquet files from {data_dir}...")
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}.")
        
    dataframes = []
    missing_cols_files = 0
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
        
        # Ensure it has all OHLCV
        expected = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not expected.issubset(set(df.collect_schema().names())):
            missing_cols_files += 1
            continue # skip this ticker completely
            
        core_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        df = df.select(core_cols).with_columns(pl.lit(ticker).alias("ticker"))
        
        cast_map = {col: pl.Float64 for col in core_cols if col != 'date'}
        df = df.cast(cast_map)
        dataframes.append(df)
        
    if missing_cols_files > 0:
        print(f"Warning: {missing_cols_files} files were missing core OHLCV columns and were skipped.")
        
    panel_df = pl.concat(dataframes, how="diagonal")
    
    # Filter to 2017 to current date
    panel_df = panel_df.filter(
        pl.col("date") >= pl.date(2017, 1, 1)
    )
    
    # Add synthetic fields
    panel_df = panel_df.sort(["ticker", "date"]).with_columns([
        (pl.col("close").pct_change()).alias("returns"),
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("vwap_proxy"),
        (pl.col("volume").rolling_mean(window_size=20)).alias("adv20"),
        (pl.col("high") - pl.col("low")).alias("range"),
        # Fill zero volumes before log to avoid -inf
        (pl.when(pl.col("volume") > 0).then(pl.col("volume").log()).otherwise(0)).alias("log_volume"),
    ])
    
    return panel_df.sort(["date", "ticker"])


# --- 3. Custom Alpha Generator ---
class AlphaGenerator:
    def __init__(self, registry: OperatorRegistry, valid_fields: List[str]):
        self.registry = registry
        self.valid_fields = valid_fields
        # Define category weights
        self.category_weights = {
            'Time Series': 0.40,
            'Cross Sectional': 0.25,
            'Arithmetic': 0.20,
            'Logical': 0.05,
            'Transformational': 0.05,
            'Vector': 0.05
        }
        
    def generate_ast(self) -> ASTNode:
        # Target 4 to 8 total operators
        target_operators = random.randint(4, 8)
        
        core_ast, _ = self._build_tree(0, 0, target_operators)
        
        # Apply neutralization optionally (Cross Sectional)
        neutral_ops = ['rank', 'normalize', 'zscore', 'winsorize', 'scale']
        if random.random() < 0.8: # 80% chance to neutralize
            op = random.choice(neutral_ops)
            core_ast = ASTNode(type="operator", value=op, children=[core_ast])
            
        # Apply smoothing optionally
        smoothing_ops = ['ts_decay_linear', 'ts_mean', 'hump']
        if random.random() < 0.8: # 80% chance to smooth
            op = random.choice(smoothing_ops)
            if op == 'hump':
                core_ast = ASTNode(type="operator", value=op, children=[core_ast])
            else:
                param = ASTNode(type="constant", value=str(random.choice([5, 10, 20, 60])))
                core_ast = ASTNode(type="operator", value=op, children=[core_ast, param])
            
        return core_ast

    def _build_tree(self, depth: int, op_count: int, target_operators: int) -> Tuple[ASTNode, int]:
        max_depth = 6
        
        if depth >= max_depth or op_count >= target_operators:
            return ASTNode(type="field", value=random.choice(self.valid_fields)), op_count
            
        # Chance to early exit if we've met min depth and ops
        if depth >= 2 and op_count >= target_operators // 2 and random.random() < 0.3:
            return ASTNode(type="field", value=random.choice(self.valid_fields)), op_count
            
        # Select an operator using registry
        op_meta = self.registry.get_random_operator(self.category_weights)
        
        children = []
        # Generate expression arguments
        for _ in range(op_meta.expr_args):
            child_node, op_count = self._build_tree(depth + 1, op_count, target_operators)
            children.append(child_node)
            
        # Generate constant arguments
        for _ in range(op_meta.const_args):
            val = str(random.randint(2, 60))
            children.append(ASTNode(type="constant", value=val))
            
        node = ASTNode(type="operator", value=op_meta.name, children=children)
        return node, op_count + 1

# --- 4. The Translator (AST -> Polars Expr) ---
def compile_to_polars(node: ASTNode) -> pl.Expr:
    if node.type == 'constant':
        try:
            val = int(node.value)
        except ValueError:
            val = float(node.value)
        return pl.lit(val)
        
    elif node.type == 'field':
        return pl.col(node.value)
    
    elif node.type == 'operator':
        op = node.value
        args = [compile_to_polars(c) for c in node.children]

        # Arithmetic
        if op == 'add': return args[0] + args[1]
        if op == 'subtract': return args[0] - args[1]
        if op == 'multiply': return args[0] * args[1]
        if op == 'divide': return args[0] / args[1]
        if op == 'power': return args[0].pow(args[1])
        if op == 'signed_power': return args[0].sign() * args[0].abs().pow(args[1])
        if op == 'sqrt': return args[0].sqrt()
        if op == 'log': return args[0].log()
        if op == 'inverse': return pl.lit(1.0) / args[0]
        if op == 'min': return pl.min_horizontal(args[0], args[1])
        if op == 'max': return pl.max_horizontal(args[0], args[1])
        if op == 'abs': return args[0].abs()
        if op == 'sign': return args[0].sign()
        if op == 'reverse': return -args[0]

        # Time Series
        if op == 'ts_mean': return args[0].rolling_mean(window_size=int(node.children[1].value))
        if op == 'ts_delay': return args[0].shift(int(node.children[1].value))
        if op == 'ts_delta': return args[0] - args[0].shift(int(node.children[1].value))
        if op == 'ts_std_dev': return args[0].rolling_std(window_size=int(node.children[1].value))
        if op == 'ts_sum': return args[0].rolling_sum(window_size=int(node.children[1].value))
        if op == 'ts_decay_linear': return args[0].rolling_mean(window_size=int(node.children[1].value))
        if op == 'ts_corr': return pl.rolling_corr(args[0], args[1], window_size=int(node.children[2].value))
        if op == 'ts_covariance': return pl.rolling_cov(args[0], args[1], window_size=int(node.children[2].value))
        if op == 'ts_rank': return args[0].rolling_map(lambda s: s.rank().tail(1).item(), window_size=int(node.children[1].value))
        if op == 'ts_scale': return (args[0] - args[0].rolling_min(window_size=int(node.children[1].value))) / (args[0].rolling_max(window_size=int(node.children[1].value)) - args[0].rolling_min(window_size=int(node.children[1].value)) + 1e-9)
        if op == 'ts_zscore': return (args[0] - args[0].rolling_mean(window_size=int(node.children[1].value))) / args[0].rolling_std(window_size=int(node.children[1].value))
        if op == 'ts_product': return (args[0].log().rolling_sum(window_size=int(node.children[1].value))).exp()
        if op == 'ts_count_nans': return args[0].is_null().cast(pl.Int32).rolling_sum(window_size=int(node.children[1].value))
        if op == 'ts_arg_max': return args[0].rolling_map(lambda s: s.arg_max(), window_size=int(node.children[1].value))
        if op == 'ts_arg_min': return args[0].rolling_map(lambda s: s.arg_min(), window_size=int(node.children[1].value))
        if op == 'ts_av_diff': return args[0] - args[0].rolling_mean(window_size=int(node.children[1].value))
        if op == 'hump': return args[0].clip(lower_bound=-0.01, upper_bound=0.01)
        
        # Cross Sectional
        if op == 'rank': return args[0].rank() / args[0].count()
        if op == 'normalize': return args[0] - args[0].mean()
        if op == 'zscore': return (args[0] - args[0].mean()) / args[0].std()
        if op == 'scale': return args[0] / args[0].abs().sum()
        if op == 'winsorize': return args[0].clip(lower_bound=args[0].mean() - 4*args[0].std(), upper_bound=args[0].mean() + 4*args[0].std())

        raise NotImplementedError(f"Operator '{op}' is not implemented in Polars compiler.")


# --- 5. Brain Simulator Metrics (Rolling Windows) ---
def _calculate_slice_metrics(df: pl.DataFrame) -> dict:
    if len(df) == 0: 
        return {"sharpe": 0.0, "turnover": 0.0, "short_pct": 0.0}

    # 1. Forward Returns (trade at T+1 close, return from T+1 close to T+2 close)
    df_eval = df.with_columns(
        pl.col("close").shift(-1).over("ticker").alias("trade_close"),
        pl.col("close").shift(-2).over("ticker").alias("exit_close")
    ).filter(
        pl.col("trade_close").is_not_null() & pl.col("exit_close").is_not_null()
    ).with_columns(
        (((pl.col("exit_close") - pl.col("trade_close")) / pl.col("trade_close")).alias("fwd_return"))
    )

    if df_eval.height == 0:
        return {"sharpe": 0.0, "turnover": 0.0, "short_pct": 0.0}

    # 2. Booksize Normalization
    try:
        df_eval = df_eval.with_columns(
            pl.col("final_alpha").abs().sum().over("date").alias("abs_sum")
        ).with_columns(
            pl.when(pl.col("abs_sum") == 0.0)
              .then(0.0)
              .otherwise(pl.col("final_alpha") / pl.col("abs_sum"))
              .alias("weight")
        ).filter(pl.col("weight").is_not_null() & pl.col("fwd_return").is_not_null())
    except Exception:
        return {"sharpe": 0.0, "turnover": 0.0, "short_pct": 0.0}

    if df_eval.height == 0:
        return {"sharpe": 0.0, "turnover": 0.0, "short_pct": 0.0}

    # 3. Portfolio Returns
    daily_stats = df_eval.group_by("date").agg(
        (pl.col("weight") * pl.col("fwd_return")).sum().alias("port_return"),
        (pl.col("weight") > 0).sum().alias("long_count"),
        (pl.col("weight") < 0).sum().alias("short_count")
    ).sort("date")

    # 4. Turnover
    turnover_df = df_eval.sort(["ticker", "date"]).with_columns(
        pl.col("weight").shift(1).over("ticker").alias("prev_weight")
    ).fill_null(0.0)

    daily_turnover = turnover_df.group_by("date").agg(
        (pl.col("weight") - pl.col("prev_weight")).abs().sum().alias("daily_turnover")
    )

    # Aggregate metrics
    mean_ret = daily_stats["port_return"].mean()
    std_ret = daily_stats["port_return"].std()
    
    if mean_ret is None or std_ret is None or std_ret == 0.0 or math.isnan(mean_ret) or math.isnan(std_ret) or math.isinf(mean_ret):
        sharpe = 0.0
    else:
        sharpe = (mean_ret / std_ret) * math.sqrt(252)

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

def calculate_rolling_metrics(df: pl.DataFrame, window_sizes: List[int] = [252, 504], step_days: int = 63) -> dict:
    """
    Evaluates alpha performance over rolling windows.
    Applies DELAY=1 to final_alpha to prevent lookahead bias.
    """
    if len(df) == 0:
        return {"median_sharpe": 0.0, "worst_sharpe": 0.0, "consistency": 0.0, "mean_turnover": 0.0, "mean_short_pct": 0.0, "windows": 0}
        
    # ENFORCE DELAY=1: Alpha calculated on T is used for trading on T+1
    df = df.sort(["ticker", "date"]).with_columns(
        pl.col("final_alpha").shift(1).over("ticker").alias("final_alpha")
    ).filter(pl.col("final_alpha").is_not_null())
    
    unique_dates = df.select("date").unique().sort("date")["date"].to_list()
    total_days = len(unique_dates)
    
    if total_days < min(window_sizes):
        return {"median_sharpe": 0.0, "worst_sharpe": 0.0, "consistency": 0.0, "mean_turnover": 0.0, "mean_short_pct": 0.0, "windows": 0}
        
    all_sharpes = []
    all_turnovers = []
    all_shorts = []
    
    for window_size in window_sizes:
        start_idx = 0
        while start_idx + window_size <= total_days:
            end_idx = start_idx + window_size
            start_date = unique_dates[start_idx]
            end_date = unique_dates[end_idx - 1]
            
            slice_df = df.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
            metrics = _calculate_slice_metrics(slice_df)
            
            all_sharpes.append(metrics["sharpe"])
            all_turnovers.append(metrics["turnover"])
            all_shorts.append(metrics["short_pct"])
            
            start_idx += step_days
            
    if not all_sharpes:
        return {"median_sharpe": 0.0, "worst_sharpe": 0.0, "consistency": 0.0, "mean_turnover": 0.0, "mean_short_pct": 0.0, "windows": 0}
        
    import statistics
    median_sharpe = statistics.median(all_sharpes)
    worst_sharpe = min(all_sharpes)
    consistency = sum(1 for s in all_sharpes if s > 0.5) / len(all_sharpes)
    mean_turnover = statistics.mean(all_turnovers)
    mean_short_pct = statistics.mean(all_shorts)
    
    return {
        "median_sharpe": median_sharpe,
        "worst_sharpe": worst_sharpe,
        "consistency": consistency * 100.0,
        "mean_turnover": mean_turnover,
        "mean_short_pct": mean_short_pct,
        "windows": len(all_sharpes)
    }

if __name__ == "__main__":
    pass