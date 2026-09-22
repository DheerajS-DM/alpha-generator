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

def run_background_runner():
    """Run the continuous generation and evaluation system."""
    print("\n" + "=" * 60)
    print("Running Alpha Background Generator (Continuous)")
    print("=" * 60)
    print("Press Ctrl+C to stop the system")
    print("=" * 60)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        result = subprocess.run(
            [sys.executable, "alpha_background_runner.py"],
            cwd=script_dir,
            capture_output=False
        )
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
        return True
    except Exception as e:
        print(f"Error running background runner: {e}")
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

def main():
    """Main entry point."""
    print("BrainQuant Generator - Main Entry Point")
    print("=" * 60)
    
    # Ask user what to run
    print("\nChoose an option:")
    print("1. Run Local Engine Test (single formula check)")
    print("2. Run Continuous Background Generator")
    print("3. Launch Web Dashboard UI")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        success = run_native_compiler()
        if success:
            print("\n✓ Engine test completed successfully")
        else:
            print("\n✗ Engine test failed")
    
    elif choice == "2":
        success = run_background_runner()
        if success:
            print("\n✓ Background generator completed successfully")
        else:
            print("\n✗ Background generator failed")
            
    elif choice == "3":
        success = run_dashboard()
        if success:
            print("\n✓ Dashboard session ended")
        else:
            print("\n✗ Dashboard failed to launch")
            
    elif choice == "4":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice. Exiting...")
        return

if __name__ == "__main__":
    main()
