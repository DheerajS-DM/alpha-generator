import unittest
import numpy as np
import polars as pl
from alpha_cache import AlphaMetricsCache, SubTreeColumnCache
from ast_node import ASTNode, parse_formula
from local_alpha_engine import compile_to_polars, materialize_subtrees

class TestAlphaCache(unittest.TestCase):
    def test_metrics_cache_basic(self):
        cache = AlphaMetricsCache(max_size=5)
        formula = "rank(ts_mean(close, 20))"
        metrics = {"median_sharpe": 1.25, "worst_sharpe": 0.5, "consistency": 80.0, "mean_turnover": 15.0, "mean_short_pct": 45.0, "windows": 10}
        
        self.assertIsNone(cache.get(formula))
        cache.set(formula, metrics)
        retrieved = cache.get(formula)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["median_sharpe"], 1.25)
        
        stats = cache.stats()
        self.assertEqual(stats["size"], 1)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)

    def test_metrics_cache_eviction(self):
        cache = AlphaMetricsCache(max_size=2)
        cache.set("f1", {"s": 1})
        cache.set("f2", {"s": 2})
        cache.set("f3", {"s": 3}) # should evict f1
        
        self.assertIsNone(cache.get("f1"))
        self.assertIsNotNone(cache.get("f2"))
        self.assertIsNotNone(cache.get("f3"))

    def test_subtree_column_cache_registration_and_hash(self):
        cache = SubTreeColumnCache(max_ram_bytes=1024 * 1024)
        sub_formula = "ts_mean(close, 20)"
        col_name = cache.hash_key(sub_formula)
        
        self.assertTrue(col_name.startswith("__st_"))
        cache.register(sub_formula, col_name)
        self.assertEqual(cache.get_column(sub_formula), col_name)

    def test_recursive_subtree_materialization(self):
        # Create small test panel dataframe
        n = 100
        df = pl.DataFrame({
            "ticker": ["AAPL"] * 50 + ["MSFT"] * 50,
            "date": list(range(50)) + list(range(50)),
            "close": np.random.uniform(100, 200, n),
            "open": np.random.uniform(100, 200, n),
            "volume": np.random.uniform(1000, 5000, n),
        })

        cache = SubTreeColumnCache(max_ram_bytes=500 * 1024 * 1024)
        node = parse_formula("rank(ts_mean(close, 10))")
        
        # Before materialization, no extra columns
        init_cols = set(df.columns)
        df_mat = materialize_subtrees(df, node, cache)
        
        # Check that ts_mean(close, 10) was materialized
        cached_col = cache.get_column("ts_mean(close, 10)")
        self.assertIsNotNone(cached_col)
        self.assertIn(cached_col, df_mat.columns)
        
        # Compiling the node with cache should reference the cached column
        compiled_expr = compile_to_polars(node, subtree_cache=cache)
        # Verify execution works with the cached expression
        res_df = df_mat.with_columns(alpha=compiled_expr.over("ticker"))
        self.assertIn("alpha", res_df.columns)
        self.assertEqual(len(res_df), n)

    def test_ram_cap_eviction(self):
        # Test that eviction triggers if memory exceeds cap
        df = pl.DataFrame({
            "ticker": ["AAPL"] * 1000,
            "col_a": [1.0] * 1000,
            "col_b": [2.0] * 1000,
        })
        # Set artificially tiny max_ram_bytes to force eviction
        cache = SubTreeColumnCache(max_ram_bytes=100) # 100 bytes is smaller than df
        cache.register("f1", "col_a")
        cache.register("f2", "col_b")
        
        df_after, evicted = cache.evict_if_needed(df)
        self.assertGreater(evicted, 0)
        self.assertNotIn("col_a", df_after.columns)

    def test_max_columns_eviction(self):
        df = pl.DataFrame({
            "ticker": ["AAPL"] * 10,
            "col_1": [1.0] * 10,
            "col_2": [2.0] * 10,
            "col_3": [3.0] * 10,
        })
        cache = SubTreeColumnCache(max_columns=2)
        cache.register("f1", "col_1")
        cache.register("f2", "col_2")
        cache.register("f3", "col_3")
        
        df_after, evicted = cache.evict_if_needed(df)
        self.assertEqual(evicted, 1)
        self.assertNotIn("col_1", df_after.columns)
        self.assertIn("col_2", df_after.columns)
        self.assertIn("col_3", df_after.columns)

if __name__ == '__main__':
    unittest.main()
