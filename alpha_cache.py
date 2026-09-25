import hashlib
import logging
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, Set
import polars as pl
from ast_node import ASTNode

logger = logging.getLogger(__name__)

class AlphaMetricsCache:
    """
    Tier 1 Cache: Formula String -> Evaluated Metrics
    Eliminates redundant evaluations when the same formula is generated
    (common in genetic evolution, mutation, and healing).
    """
    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, formula: str) -> Optional[Dict[str, Any]]:
        if formula in self._cache:
            self._cache.move_to_end(formula)
            self.hits += 1
            return self._cache[formula]
        self.misses += 1
        return None

    def set(self, formula: str, metrics: Dict[str, Any]):
        if formula in self._cache:
            self._cache.move_to_end(formula)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[formula] = metrics

    def preload_from_csv(self, csv_path: str) -> int:
        """Pre-populate the cache from historical logs_processed.csv."""
        import os
        import csv
        if not os.path.exists(csv_path):
            return 0
        loaded = 0
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    formula = row.get("formula")
                    if not formula:
                        continue
                    try:
                        metrics = {
                            "median_sharpe": float(row.get("median_sharpe", 0.0)),
                            "worst_sharpe": float(row.get("worst_sharpe", 0.0)),
                            "consistency": float(row.get("consistency", 0.0)),
                            "mean_turnover": float(row.get("turnover", 0.0)),
                            "mean_short_pct": float(row.get("short_pct", 0.0)),
                            "windows": int(float(row.get("windows", 0)))
                        }
                        self.set(formula, metrics)
                        loaded += 1
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Failed to preload metrics cache from {csv_path}: {e}")
        return loaded

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100.0) if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 2)
        }


class SubTreeColumnCache:
    """
    Tier 2 Cache: Recursive Sub-Tree Materialization & Column Reuse in DataFrame.
    Capped at max_columns (default 30) and max_ram_bytes (default 2GB).
    When expressions share common subtrees (e.g. ts_mean(close, 20), ts_zscore(close, 60)),
    the computed Series is retained as a cached column in the DataFrame and looked up via pl.col().
    """
    def __init__(self, max_ram_bytes: int = 2 * 1024 * 1024 * 1024, max_columns: int = 30):
        self.max_ram_bytes = max_ram_bytes
        self.max_columns = max_columns
        # Map formula string -> internal column name in df
        self.formula_to_col: OrderedDict[str, str] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def hash_key(formula: str) -> str:
        digest = hashlib.md5(formula.encode('utf-8')).hexdigest()[:12]
        return f"__st_{digest}"

    def get_column(self, formula: str) -> Optional[str]:
        if formula in self.formula_to_col:
            self.formula_to_col.move_to_end(formula)
            self.hits += 1
            return self.formula_to_col[formula]
        self.misses += 1
        return None

    def register(self, formula: str, col_name: str):
        if formula in self.formula_to_col:
            self.formula_to_col.move_to_end(formula)
        else:
            self.formula_to_col[formula] = col_name

    def clear(self):
        """Clears all cached column mappings."""
        self.formula_to_col.clear()

    def evict_if_needed(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, int]:
        """
        Enforce max_columns and max_ram_bytes constraint.
        Evicts the oldest cached subtrees from both the registry and the DataFrame.
        """
        evicted = 0
        cols_to_drop = []

        # 1. Enforce hard column count cap (prevents wide-table Polars slowdown)
        while len(self.formula_to_col) > self.max_columns:
            _, old_col = self.formula_to_col.popitem(last=False)
            if old_col in df.columns:
                cols_to_drop.append(old_col)
            evicted += 1
            self.evictions += 1

        # 2. Enforce RAM cap
        try:
            curr_size = df.estimated_size()
            while curr_size > self.max_ram_bytes and self.formula_to_col:
                _, old_col = self.formula_to_col.popitem(last=False)
                if old_col in df.columns:
                    cols_to_drop.append(old_col)
                evicted += 1
                self.evictions += 1
                if cols_to_drop:
                    df = df.drop(cols_to_drop)
                    cols_to_drop = []
                curr_size = df.estimated_size()
        except Exception:
            pass

        if cols_to_drop:
            df = df.drop(cols_to_drop)

        return df, evicted

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100.0) if total > 0 else 0.0
        return {
            "cached_subtrees": len(self.formula_to_col),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 2),
            "evictions": self.evictions
        }
