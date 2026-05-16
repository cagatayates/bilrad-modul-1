#!/usr/bin/env python3
"""
Example usage of result comparison script
"""

import subprocess
import sys

def run_comparison_example():
    """Run result comparison with example parameters"""
    
    print("Example: Result Comparison Analysis")
    print("="*60)
    print("This example shows how to compare results between main and denoised datasets.")
    print("The script will:")
    print("1. Load results from two CSV files")
    print("2. Sort main results by SNR (ascending)")
    print("3. Extract frame IDs from specified index range")
    print("4. Find corresponding metrics in denoised results")
    print("5. Calculate and display average Dice Score and SNR")
    print()
    
    # Example command for result comparison
    cmd = [
        "python", "result_compare.py",
        "--main-results", "results_test_1_main.csv",
        "--denoised-results", "results_test_1_denoised.csv",
        "--start-index", "0",
        "--end-index", "50",
        "--verbose"
    ]
    
    print("Example command:")
    print(" ".join(cmd))
    print("\n" + "="*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
    except FileNotFoundError:
        print("Error: Python or result_compare.py not found. Make sure you're in the correct directory.")

if __name__ == "__main__":
    run_comparison_example()
