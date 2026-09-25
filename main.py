"""
Main entry point for the BrainQuant Generator system.
"""

import os
import sys
import subprocess
import time

def run_native_compiler():
    """Run the native compiler test."""
    print("=" * 60)
    print("Running Native Compiler Test")
    print("=" * 60)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        result = subprocess.run(
            [sys.executable, "local_alpha_engine.py"],
            cwd=script_dir,
            capture_output=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running native compiler: {e}")
        return False

def run_background_runner(strategy="hybrid"):
    """Run the continuous generation and evaluation system with a chosen strategy."""
    strategy_names = {
        "hybrid": "Multi-Strategy Hybrid (Templates + Evolver + Healer)",
        "templates": "Template-Based Tree Generator (Market Scenarios & Dips)",
        "evolve": "Genetic Evolver (AST Crossover & Mutation)",
        "heal": "Alpha Healer (Targeted Near-Miss Fixes)",
        "random": "Legacy Random Generator"
    }
    title = strategy_names.get(strategy, strategy)
    print("\n" + "=" * 60)
    print(f"Running Alpha Generator: {title}")
    print("=" * 60)
    print("Press Ctrl+C to stop the system")
    print("=" * 60)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        result = subprocess.run(
            [sys.executable, "alpha_background_runner.py", "--strategy", strategy],
            cwd=script_dir,
            capture_output=False
        )
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
        return True
    except Exception as e:
        print(f"Error running background runner ({strategy}): {e}")
        return False

def run_dashboard():
    """Launch the Web Dashboard."""
    print("\n" + "=" * 60)
    print("Launching BrainQuant Web Dashboard")
    print("=" * 60)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        result = subprocess.run(
            [sys.executable, "dashboard.py"],
            cwd=script_dir,
            capture_output=False
        )
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
        return True
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        return False

def run_benchmark(num_alphas: int = 30):
    """Run the benchmark evaluator to measure processing and latency."""
    print("\n" + "=" * 60)
    print(f"Running Benchmark Evaluator ({num_alphas} Alphas)")
    print("=" * 60)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        result = subprocess.run(
            [sys.executable, "benchmark_observer.py", str(num_alphas)],
            cwd=script_dir,
            capture_output=False
        )
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nBenchmark stopped by user")
        return True
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return False

def main():
    """Main entry point."""
    print("=" * 60)
    print("       BrainQuant Generator - Quantitative Research System       ")
    print("=" * 60)
    
    print("\nChoose an option:")
    print("1. Run Local Engine Test (single formula check)")
    print("2. Run Continuous Alpha Discovery (Hybrid: Templates + Evolution + Healing)")
    print("3. Run Template-Based Generator (Market Scenarios & Dip-Buying)")
    print("4. Run Genetic Evolver (AST Crossover & Mutation on Elite Pool)")
    print("5. Run Alpha Healer (Targeted Fixes on Near-Miss Alphas)")
    print("6. Run Legacy Random Generator")
    print("7. Launch Web Dashboard UI")
    print("8. Run Processing & Latency Benchmark Evaluator")
    print("9. Exit")
    
    choice = input("\nEnter choice (1-9): ").strip()
    
    if choice == "1":
        success = run_native_compiler()
        if success:
            print("\n✓ Engine test completed successfully")
        else:
            print("\n✗ Engine test failed")
    
    elif choice == "2":
        run_background_runner(strategy="hybrid")
            
    elif choice == "3":
        run_background_runner(strategy="templates")

    elif choice == "4":
        run_background_runner(strategy="evolve")

    elif choice == "5":
        run_background_runner(strategy="heal")

    elif choice == "6":
        run_background_runner(strategy="random")

    elif choice == "7":
        success = run_dashboard()
        if success:
            print("\n✓ Dashboard session ended")
        else:
            print("\n✗ Dashboard failed to launch")

    elif choice == "8":
        success = run_benchmark()
        if success:
            print("\n✓ Benchmark evaluation completed successfully")
        else:
            print("\n✗ Benchmark evaluation failed")
            
    elif choice == "9":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice. Exiting...")
        return

if __name__ == "__main__":
    main()
