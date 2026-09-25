import unittest
import numpy as np
import polars as pl
from ast_node import parse_formula
from local_alpha_engine import compile_to_polars, calculate_rolling_metrics, _calculate_slice_metrics

class TestDollarNeutrality(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 10-stock, 300-day panel dataset
        np.random.seed(42)
        n_tickers = 10
        n_days = 300
        tickers = [f"TICK_{i}" for i in range(n_tickers)]
        
        dates = []
        ticker_list = []
        closes = []
        highs = []
        lows = []
        opens = []
        volumes = []

        for d in range(n_days):
            for t in tickers:
                dates.append(d)
                ticker_list.append(t)
                base = 100.0 + np.random.uniform(0, 50) + (d * 0.1) # Upward trending market
                closes.append(base)
                highs.append(base * 1.02)
                lows.append(base * 0.98)
                opens.append(base * 0.99)
                volumes.append(10000 + np.random.randint(0, 5000))

        self.df = pl.DataFrame({
            "ticker": ticker_list,
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }).sort(["ticker", "date"])

    def test_strict_dollar_neutral_weights(self):
        # Even if the raw signal is strictly positive (e.g. close price which is > 0)
        # the weights must sum to approximately 0.0 per day (dollar neutrality)
        raw_df = self.df.with_columns(final_alpha=pl.col("close"))
        metrics = _calculate_slice_metrics(raw_df)
        
        # Short percentage should be ~50% because weights are demeaned
        self.assertGreaterEqual(metrics["short_pct"], 35.0)
        self.assertLessEqual(metrics["short_pct"], 65.0)

    def test_time_series_smoothing_after_zscore_neutrality(self):
        # Formula: ts_decay_linear(reverse(zscore(high)), 5)
        # Even though ts_decay_linear smoothes across time, the slice evaluation demeans final_alpha
        node = parse_formula("ts_decay_linear(reverse(zscore(high)), 5)")
        expr = compile_to_polars(node)
        
        df_sig = (
            self.df
            .with_columns(alpha_signal=expr.over("ticker"))
            .with_columns(final_alpha=pl.col("alpha_signal").over("date"))
        )
        
        metrics = _calculate_slice_metrics(df_sig)
        self.assertGreaterEqual(metrics["short_pct"], 35.0)
        self.assertLessEqual(metrics["short_pct"], 65.0)

if __name__ == '__main__':
    unittest.main()
