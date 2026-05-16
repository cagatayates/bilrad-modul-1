#!/usr/bin/env python3
"""
Example usage of batch evaluation functionality with SNR calculation and CSV export
"""

import subprocess
import sys

def run_batch_evaluation():
    """Run batch evaluation with example parameters"""
    
    print("Example 1: Batch evaluation without SNR")
    print("="*60)
    
    # Example command for batch evaluation without SNR
    cmd = [
        "python", "inference.py",
        "--checkpoint", "checkpoints/unet1d_pit_case1_N3_best.pth",
        "--data", "data/deinterleaving_u_net_case_1_2_scenario_data.npy",
        "--labels", "data/deinterleaving_u_net_case_1_2_scenario_labels.npy",
        "--batch-eval",
        "--start-frame", "0",
        "--end-frame", "10",
        "--threshold", "0.5"
    ]
    
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
        print("Error: Python or inference.py not found. Make sure you're in the correct directory.")
    
    print("\n" + "="*60)
    print("Example 2: Batch evaluation with SNR calculation")
    print("="*60)
    
    # Example command for batch evaluation with SNR
    cmd_snr = [
        "python", "inference.py",
        "--checkpoint", "checkpoints/unet1d_pit_case1_N3_best.pth",
        "--data", "data/deinterleaving_u_net_test_1_verici_2_scenario_data.npy",
        "--labels", "data/deinterleaving_u_net_test_1_verici_2_scenario_labels.npy",
        "--batch-eval",
        "--start-frame", "0",
        "--end-frame", "10",
        "--threshold", "0.5",
        "--enable-snr",
        "--ground-truth", "data/deinterleaving_u_net_test_1_verici_2_ground_truth_data.npy"
    ]
    
    print(" ".join(cmd_snr))
    print("\n" + "="*60)
    
    try:
        result = subprocess.run(cmd_snr, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
    except FileNotFoundError:
        print("Error: Python or inference.py not found. Make sure you're in the correct directory.")
    
    print("\n" + "="*60)
    print("Example 3: Batch evaluation with SNR and CSV export")
    print("="*60)
    
    # Example command for batch evaluation with SNR and CSV export
    cmd_csv = [
        "python", "inference.py",
        "--checkpoint", "checkpoints/unet1d_pit_case1_N3_best.pth",
        "--data", "data/deinterleaving_u_net_test_1_verici_2_scenario_data.npy",
        "--labels", "data/deinterleaving_u_net_test_1_verici_2_scenario_labels.npy",
        "--batch-eval",
        "--start-frame", "0",
        "--end-frame", "10",
        "--threshold", "0.5",
        "--enable-snr",
        "--ground-truth", "data/deinterleaving_u_net_test_1_verici_2_ground_truth_data.npy",
        "--save-csv", "results_frame_metrics.csv"
    ]
    
    print(" ".join(cmd_csv))
    print("\n" + "="*60)
    
    try:
        result = subprocess.run(cmd_csv, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
    except FileNotFoundError:
        print("Error: Python or inference.py not found. Make sure you're in the correct directory.")

if __name__ == "__main__":
    print("Example: Batch Evaluation with SNR and CSV Export")
    print("="*60)
    print("This example shows how to run batch evaluation on frames 0-9")
    print("with the trained model and calculate average performance metrics.")
    print("The examples include SNR calculation and CSV export functionality.")
    print()
    
    run_batch_evaluation()