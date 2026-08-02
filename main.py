"""
Main entry point for the brainquantgenerator system.
Runs both the native compiler test and the hybrid translation system.
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

def run_hybrid_system():
    """Run the hybrid translation system."""
    print("\n" + "=" * 60)
    print("Running Hybrid Translation System")
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
        print(f"Error running hybrid system: {e}")
        return False

def main():
    """Main entry point."""
    print("BrainQuant Generator - Main Entry Point")
    print("=" * 60)
    
    # Ask user what to run
    print("\nChoose an option:")
    print("1. Run Native Compiler Test (single formula)")
    print("2. Run Hybrid Translation System (continuous)")
    print("3. Run Both (Native test first, then Hybrid)")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        success = run_native_compiler()
        if success:
            print("\n✓ Native compiler test completed successfully")
        else:
            print("\n✗ Native compiler test failed")
    
    elif choice == "2":
        success = run_hybrid_system()
        if success:
            print("\n✓ Hybrid system completed successfully")
        else:
            print("\n✗ Hybrid system failed")
    
    elif choice == "3":
        print("\n--- Step 1: Native Compiler Test ---")
        success1 = run_native_compiler()
        
        time.sleep(2)
        
        print("\n--- Step 2: Hybrid Translation System ---")
        success2 = run_hybrid_system()
        
        if success1 and success2:
            print("\n✓ Both systems completed successfully")
        else:
            print(f"\n✗ System status: Native={success1}, Hybrid={success2}")
    
    elif choice == "4":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice. Exiting...")
        return

if __name__ == "__main__":
    main()
