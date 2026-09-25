import os
import glob
import random
import json
import polars as pl
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math
import numpy as np
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
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("vwap"),
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("vwap_proxy"),
        (pl.col("volume").rolling_mean(window_size=20)).alias("adv20"),
        (pl.col("high") - pl.col("low")).alias("range"),
        # Fill zero volumes before log to avoid -inf
        (pl.when(pl.col("volume") > 0).then(pl.col("volume").log()).otherwise(0)).alias("log_volume"),
    ])
    
    return panel_df.sort(["date", "ticker"])


def split_data_temporal(df: pl.DataFrame, train_pct: float = 0.70):
    """
    Split data into in-sample (IS) and out-of-sample (OOS) by date.
    
    This addresses the fundamental data-mining bias: when searching thousands
    of formulas on the same dataset, some will appear excellent purely by chance.
    By discovering alphas on IS data and validating on unseen OOS data,
    we can distinguish genuine predictive signal from overfitting.
    
    Args:
        df: Full dataset (must have a 'date' column)
        train_pct: Fraction of dates for in-sample discovery (default 70%)
        
    Returns:
        (is_df, oos_df, split_date) — the two DataFrames and the cutoff date
    """
    unique_dates = df.select("date").unique().sort("date")["date"].to_list()
    split_idx = int(len(unique_dates) * train_pct)
    split_date = unique_dates[split_idx]
    
    is_df = df.filter(pl.col("date") < split_date)
    oos_df = df.filter(pl.col("date") >= split_date)
    
    return is_df, oos_df, split_date


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

TIME_SERIES_OPS = {
    'ts_mean', 'ts_delay', 'ts_delta', 'ts_std_dev', 'ts_sum', 'ts_decay_linear',
    'ts_corr', 'ts_covariance', 'ts_rank', 'ts_scale', 'ts_zscore', 'ts_product',
    'ts_count_nans', 'ts_arg_max', 'ts_arg_min', 'ts_av_diff'
}
CROSS_SECTIONAL_OPS = {'rank', 'normalize', 'zscore', 'scale', 'winsorize', 'quantile'}


def evaluate_ast(df: pl.DataFrame, node: ASTNode, subtree_cache=None) -> tuple[pl.DataFrame, pl.Expr]:
    """
    Evaluates an AST expression tree on df with strict dimensional parity:
    - Time-series operators (ts_*) are evaluated strictly over('ticker') along chronological order.
    - Cross-sectional operators (rank, zscore, normalize, scale, winsorize) are evaluated strictly over('date') across all tickers on each date.
    - Arithmetic operators operate element-wise.
    
    Eliminates lookahead bias by ensuring cross-sectional operators never aggregate across future dates.
    """
    temp_counter = [0]
    
    def _eval_node(current_df: pl.DataFrame, n: ASTNode) -> tuple[pl.DataFrame, pl.Expr]:
        # Check subtree cache first
        if subtree_cache and n.type == 'operator':
            formula_str = n.to_string()
            cached_col = subtree_cache.get_column(formula_str)
            if cached_col and cached_col in current_df.columns:
                return current_df, pl.col(cached_col)
                
        if n.type == 'constant':
            try:
                val = int(n.value)
            except ValueError:
                val = float(n.value)
            return current_df, pl.lit(val)
            
        elif n.type == 'field':
            return current_df, pl.col(n.value)
            
        elif n.type == 'operator':
            op = n.value
            child_exprs = []
            for child in (n.children or []):
                current_df, c_expr = _eval_node(current_df, child)
                child_exprs.append(c_expr)
                
            if op == 'add': expr = child_exprs[0] + child_exprs[1]
            elif op == 'subtract': expr = child_exprs[0] - child_exprs[1]
            elif op == 'multiply': expr = child_exprs[0] * child_exprs[1]
            elif op == 'divide': expr = child_exprs[0] / child_exprs[1]
            elif op == 'power': expr = child_exprs[0].pow(child_exprs[1])
            elif op == 'signed_power': expr = child_exprs[0].sign() * child_exprs[0].abs().pow(child_exprs[1])
            elif op == 'sqrt': expr = child_exprs[0].sqrt()
            elif op == 'log': expr = child_exprs[0].log()
            elif op == 'inverse': expr = pl.lit(1.0) / child_exprs[0]
            elif op == 'min': expr = pl.min_horizontal(child_exprs[0], child_exprs[1])
            elif op == 'max': expr = pl.max_horizontal(child_exprs[0], child_exprs[1])
            elif op == 'abs': expr = child_exprs[0].abs()
            elif op == 'sign': expr = child_exprs[0].sign()
            elif op == 'reverse': expr = -child_exprs[0]
            elif op == 'hump': expr = child_exprs[0] / (1.0 + child_exprs[0].abs())
            elif op == 'if_else':
                expr = pl.when(child_exprs[0] != 0).then(child_exprs[1]).otherwise(child_exprs[2])
            elif op in TIME_SERIES_OPS:
                if op in ('ts_corr', 'ts_covariance'):
                    w = int(n.children[2].value) if len(n.children) > 2 else 10
                else:
                    w = int(n.children[1].value) if len(n.children) > 1 else 10
                if op == 'ts_mean': ts_expr = child_exprs[0].rolling_mean(window_size=w)
                elif op == 'ts_delay': ts_expr = child_exprs[0].shift(w)
                elif op == 'ts_delta': ts_expr = child_exprs[0] - child_exprs[0].shift(w)
                elif op == 'ts_std_dev': ts_expr = child_exprs[0].rolling_std(window_size=w)
                elif op == 'ts_sum': ts_expr = child_exprs[0].rolling_sum(window_size=w)
                elif op == 'ts_decay_linear':
                    w_sum = (w * (w + 1)) / 2.0
                    terms = [(w - i) * child_exprs[0].shift(i) for i in range(w)]
                    ts_expr = pl.sum_horizontal(terms) / w_sum
                elif op == 'ts_corr':
                    ts_expr = pl.rolling_corr(child_exprs[0], child_exprs[1], window_size=int(n.children[2].value))
                elif op == 'ts_covariance':
                    ts_expr = pl.rolling_cov(child_exprs[0], child_exprs[1], window_size=int(n.children[2].value))
                elif op == 'ts_rank':
                    ts_expr = child_exprs[0].rolling_map(lambda s: s.rank().tail(1).item(), window_size=w)
                elif op == 'ts_scale':
                    ts_expr = (child_exprs[0] - child_exprs[0].rolling_min(window_size=w)) / (child_exprs[0].rolling_max(window_size=w) - child_exprs[0].rolling_min(window_size=w) + 1e-9)
                elif op == 'ts_zscore':
                    ts_expr = (child_exprs[0] - child_exprs[0].rolling_mean(window_size=w)) / child_exprs[0].rolling_std(window_size=w)
                elif op == 'ts_product':
                    ts_expr = (child_exprs[0].log().rolling_sum(window_size=w)).exp()
                elif op == 'ts_count_nans':
                    ts_expr = child_exprs[0].is_null().cast(pl.Int32).rolling_sum(window_size=w)
                elif op == 'ts_arg_max':
                    ts_expr = child_exprs[0].rolling_map(lambda s: s.arg_max(), window_size=w)
                elif op == 'ts_arg_min':
                    ts_expr = child_exprs[0].rolling_map(lambda s: s.arg_min(), window_size=w)
                elif op == 'ts_av_diff':
                    ts_expr = child_exprs[0] - child_exprs[0].rolling_mean(window_size=w)
                else:
                    ts_expr = child_exprs[0].rolling_mean(window_size=w)

                col_name = f"__ts_{temp_counter[0]}"
                temp_counter[0] += 1
                current_df = current_df.with_columns(ts_expr.over("ticker").alias(col_name))
                if subtree_cache and n.type == 'operator':
                    subtree_cache.register(n.to_string(), col_name)
                    current_df, _ = subtree_cache.evict_if_needed(current_df)
                return current_df, pl.col(col_name)

            elif op in CROSS_SECTIONAL_OPS:
                if op == 'rank': cs_expr = child_exprs[0].rank() / child_exprs[0].count()
                elif op == 'normalize': cs_expr = child_exprs[0] - child_exprs[0].mean()
                elif op == 'zscore': cs_expr = (child_exprs[0] - child_exprs[0].mean()) / child_exprs[0].std()
                elif op == 'scale': cs_expr = child_exprs[0] / child_exprs[0].abs().sum()
                elif op == 'winsorize': cs_expr = child_exprs[0].clip(child_exprs[0].mean() - 4*child_exprs[0].std(), child_exprs[0].mean() + 4*child_exprs[0].std())
                else: cs_expr = child_exprs[0].rank() / child_exprs[0].count()

                col_name = f"__cs_{temp_counter[0]}"
                temp_counter[0] += 1
                current_df = current_df.with_columns(cs_expr.over("date").alias(col_name))
                if subtree_cache and n.type == 'operator':
                    subtree_cache.register(n.to_string(), col_name)
                    current_df, _ = subtree_cache.evict_if_needed(current_df)
                return current_df, pl.col(col_name)

            else:
                raise NotImplementedError(f"Operator '{op}' is not implemented in Polars compiler.")

            return current_df, expr
            
        raise ValueError(f"Unknown ASTNode type: {n.type}")

    return _eval_node(df, node)


# --- 4. The Translator (AST -> Polars Expr) ---
def compile_to_polars(node: ASTNode, subtree_cache=None) -> pl.Expr:
    # If subtree cache is provided and this node is already cached as a column in the DataFrame
    if subtree_cache is not None and node.type == 'operator':
        cached_col = subtree_cache.get_column(node.to_string())
        if cached_col:
            return pl.col(cached_col)

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
        args = [compile_to_polars(c, subtree_cache=subtree_cache) for c in node.children]

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
        if op == 'ts_decay_linear':
            window = int(node.children[1].value)
            w_sum = (window * (window + 1)) / 2.0
            terms = [(window - i) * args[0].shift(i) for i in range(window)]
            return pl.sum_horizontal(terms) / w_sum
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
        if op == 'hump': return args[0] / (1.0 + args[0].abs())
        
        # Cross Sectional
        if op == 'rank': return args[0].rank() / args[0].count()
        if op == 'normalize': return args[0] - args[0].mean()
        if op == 'zscore': return (args[0] - args[0].mean()) / args[0].std()
        if op == 'scale': return args[0] / args[0].abs().sum()
        if op == 'winsorize': return args[0].clip(lower_bound=args[0].mean() - 4*args[0].std(), upper_bound=args[0].mean() + 4*args[0].std())

        raise NotImplementedError(f"Operator '{op}' is not implemented in Polars compiler.")


def materialize_subtrees(df: pl.DataFrame, node: ASTNode, subtree_cache, max_depth: int = 4) -> pl.DataFrame:
    """
    Recursively finds eligible time-series / complex sub-trees inside node,
    evaluates them against df if not already cached, and appends them as columns.
    Enforces the cache's RAM cap via LRU eviction.
    """
    if subtree_cache is None or node.type != 'operator':
        return df

    # We target caching operator nodes that perform substantial work (like time-series or multi-node arithmetic)
    # Recursively check children first (bottom-up materialization)
    for child in (node.children or []):
        if child.type == 'operator':
            df = materialize_subtrees(df, child, subtree_cache, max_depth=max_depth)

    # Don't cache the root top-level operator itself here (as that will become alpha_signal)
    formula_str = node.to_string()
    if node.value in ('ts_mean', 'ts_std_dev', 'ts_decay_linear', 'ts_zscore', 'ts_corr', 'ts_scale', 'ts_delta', 'ts_sum'):
        cached_col = subtree_cache.get_column(formula_str)
        if not cached_col or cached_col not in df.columns:
            col_name = subtree_cache.hash_key(formula_str)
            try:
                # Compile using cached children if available
                expr = compile_to_polars(node, subtree_cache=subtree_cache)
                df = df.with_columns(expr.over("ticker").alias(col_name))
                subtree_cache.register(formula_str, col_name)
                # Enforce memory cap (e.g. 4GB)
                df, evicted = subtree_cache.evict_if_needed(df)
            except Exception:
                pass

    return df



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

    # 2. Strict Dollar Neutralization & Booksize Normalization
    try:
        df_eval = df_eval.with_columns(
            (pl.col("final_alpha") - pl.col("final_alpha").mean().over("date")).alias("neutral_alpha")
        ).with_columns(
            pl.col("neutral_alpha").abs().sum().over("date").alias("abs_sum")
        ).with_columns(
            pl.when(pl.col("abs_sum") == 0.0)
              .then(0.0)
              .otherwise(pl.col("neutral_alpha") / pl.col("abs_sum"))
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
    Delay 1 is accurately modeled: Alpha generated at close of day T trades at
    close of day T+1, realizing return from T+1 close to T+2 close.
    """
    if len(df) == 0:
        return {"median_sharpe": 0.0, "worst_sharpe": 0.0, "consistency": 0.0, "mean_turnover": 0.0, "mean_short_pct": 0.0, "windows": 0}
        
    df = df.sort(["ticker", "date"]).filter(pl.col("final_alpha").is_not_null())
    
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