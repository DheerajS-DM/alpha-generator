"""
BrainQuant Benchmark Observer
Monitors and measures throughput, latencies, memory consumption,
and cache efficiency across alpha generation and evaluation phases.
Outputs live metrics to benchmark_metrics.json for dashboard consumption.
"""

import os
import sys
import time
import json
import ctypes
from typing import Dict, Any, List

METRICS_FILE = "benchmark_metrics.json"

def _get_process_rss_mb() -> float:
    """Retrieve process RSS memory in MB using ctypes on Windows."""
    if os.name == 'nt':
        try:
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ('cb', ctypes.c_ulong),
                    ('PageFaultCount', ctypes.c_ulong),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t),
                ]
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            
            get_mem_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_mem_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_ulong]
            get_mem_info.restype = ctypes.c_int
            
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if get_mem_info(handle, ctypes.byref(counters), counters.cb):
                return round(counters.WorkingSetSize / (1024 * 1024), 2)
        except Exception:
            pass
    return 0.0

class BenchmarkObserver:
    def __init__(self, output_path: str = METRICS_FILE):
        self.output_path = output_path
        
        # Cumulative and rolling latency tracking
        self.latencies: Dict[str, List[float]] = {
            "ast_generation_ms": [],
            "compiler_ms": [],
            "polars_eval_ms": [],
            "rolling_metrics_ms": [],
            "total_alpha_ms": []
        }
        self.max_history = 200
        
        # Counts
        self.total_evaluated = 0
        self.total_cache_hits_t1 = 0
        self.total_cache_hits_t2 = 0
        self.start_time = time.time()

    def record_stage(self, stage: str, duration_sec: float):
        ms = duration_sec * 1000.0
        if stage in self.latencies:
            self.latencies[stage].append(ms)
            if len(self.latencies[stage]) > self.max_history:
                self.latencies[stage].pop(0)

    def record_evaluation(self, duration_sec: float, hit_tier1: bool = False, hit_tier2: bool = False):
        self.total_evaluated += 1
        if hit_tier1:
            self.total_cache_hits_t1 += 1
        if hit_tier2:
            self.total_cache_hits_t2 += 1
        self.record_stage("total_alpha_ms", duration_sec)

    def get_summary(self) -> Dict[str, Any]:
        elapsed = max(0.001, time.time() - self.start_time)
        throughput_per_sec = self.total_evaluated / elapsed
        
        # Memory metrics
        rss_mb = _get_process_rss_mb()
        
        stats: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "uptime_seconds": round(elapsed, 1),
            "total_evaluated": self.total_evaluated,
            "throughput_alphas_per_sec": round(throughput_per_sec, 2),
            "process_rss_mb": round(rss_mb, 2),
            "tier1_cache_hits": self.total_cache_hits_t1,
            "tier2_cache_hits": self.total_cache_hits_t2,
            "latencies_ms": {}
        }

        for stage, samples in self.latencies.items():
            if samples:
                stats["latencies_ms"][stage] = {
                    "avg": round(sum(samples) / len(samples), 2),
                    "min": round(min(samples), 2),
                    "max": round(max(samples), 2),
                    "p95": round(float(sorted(samples)[int(len(samples) * 0.95)]), 2) if len(samples) >= 10 else round(samples[-1], 2)
                }
            else:
                stats["latencies_ms"][stage] = {"avg": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}

        return stats

    def flush_to_disk(self):
        summary = self.get_summary()
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass
        return summary


def run_standalone_benchmark(num_alphas: int = 50):
    """
    Evaluates a sample batch of alphas using the native pipeline and measures
    exact component-level latency and processing throughput.
    """
    print("=" * 65)
    print(f"BrainQuant Benchmark Evaluator: Running {num_alphas} Alphas")
    print("=" * 65)

    from operator_registry import OperatorRegistry
    from local_alpha_engine import load_local_data, compile_to_polars, calculate_rolling_metrics, split_data_temporal, materialize_subtrees
    from alpha_templates import AlphaTemplateGenerator
    from alpha_cache import AlphaMetricsCache, SubTreeColumnCache
    import polars as pl

    observer = BenchmarkObserver()
    registry = OperatorRegistry("operatorRAW.json")
    fields = ["open", "high", "low", "close", "volume"]
    tpl_gen = AlphaTemplateGenerator(valid_fields=fields)

    print("Loading in-sample data...")
    t0 = time.time()
    df_eager = load_local_data().collect()
    is_df, oos_df, _ = split_data_temporal(df_eager, train_pct=0.70)
    print(f"Data loaded: {is_df.height:,} rows in {time.time() - t0:.2f}s\n")

    metrics_cache = AlphaMetricsCache()
    subtree_cache = SubTreeColumnCache(max_ram_bytes=4 * 1024 * 1024 * 1024)

    print(f"Evaluating {num_alphas} candidate alphas across all stages...")
    for i in range(1, num_alphas + 1):
        alpha_start = time.time()
        
        # Stage 1: Generation
        t_gen0 = time.time()
        node = tpl_gen.generate()
        formula_str = node.to_string()
        observer.record_stage("ast_generation_ms", time.time() - t_gen0)

        # Tier 1 cache check
        if metrics_cache.get(formula_str):
            observer.record_evaluation(time.time() - alpha_start, hit_tier1=True)
            continue

        # Stage 2: Sub-tree materialization & compilation
        t_comp0 = time.time()
        is_df = materialize_subtrees(is_df, node, subtree_cache)
        expr = compile_to_polars(node, subtree_cache=subtree_cache)
        observer.record_stage("compiler_ms", time.time() - t_comp0)

        # Stage 3: Polars Evaluation
        t_polars0 = time.time()
        is_tmp = (
            is_df
            .with_columns(alpha_signal=expr.over("ticker"))
            .with_columns(final_alpha=pl.col("alpha_signal").over("date"))
        )
        observer.record_stage("polars_eval_ms", time.time() - t_polars0)

        # Stage 4: Rolling Backtest Metrics
        t_metrics0 = time.time()
        metrics = calculate_rolling_metrics(is_tmp, window_sizes=[252], step_days=63)
        metrics_cache.set(formula_str, metrics)
        observer.record_stage("rolling_metrics_ms", time.time() - t_metrics0)

        observer.record_evaluation(time.time() - alpha_start)

        if i % 10 == 0 or i == num_alphas:
            summary = observer.flush_to_disk()
            print(f"Progress: {i}/{num_alphas} | Throughput: {summary['throughput_alphas_per_sec']} alphas/sec | RAM: {summary['process_rss_mb']} MB")

    final_report = observer.flush_to_disk()
    print("\n" + "=" * 65)
    print("BENCHMARK EVALUATION COMPLETE")
    print("=" * 65)
    print(f"Total Evaluated: {final_report['total_evaluated']}")
    print(f"Throughput:      {final_report['throughput_alphas_per_sec']} alphas/sec")
    print(f"Process RAM:     {final_report['process_rss_mb']} MB")
    print("\nLatency Breakdown (avg ms):")
    for k, v in final_report["latencies_ms"].items():
        print(f"  - {k:<22} Avg: {v['avg']:>6.2f} ms | P95: {v['p95']:>6.2f} ms | Max: {v['max']:>6.2f} ms")
    print(f"\nLive stats saved to {METRICS_FILE}")
    return final_report

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run_standalone_benchmark(count)
