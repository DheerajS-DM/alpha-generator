"""
Benchmark comparison between vectorized Polars ts_decay_linear vs rolling_map.
"""
import unittest
import time
import numpy as np
import polars as pl

class TestPerformance(unittest.TestCase):
    def test_ts_decay_linear_vectorized(self):
        w = 10
        n_rows = 100_000
        data = np.random.randn(n_rows)
        df = pl.DataFrame({"x": data})

        # Vectorized implementation: sum_i (w - i) * shift(i) / sum(weights)
        t0 = time.time()
        w_sum = (w * (w + 1)) // 2
        terms = [(w - i) * pl.col("x").shift(i) for i in range(w)]
        expr = pl.sum_horizontal(terms) / float(w_sum)
        res = df.select(expr).to_series()
        t_vec = time.time() - t0

        print(f"\nVectorized ts_decay_linear({w}) on {n_rows:,} rows took: {t_vec*1000:.2f} ms")
        self.assertEqual(len(res), n_rows)
        self.assertLess(t_vec, 1.0, "Vectorized calculation should complete in under 1 second")

if __name__ == '__main__':
    unittest.main()
