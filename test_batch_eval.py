#!/usr/bin/env python3
"""
Test script for batch evaluation functionality
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import RadarDeinterleavingInference

def test_batch_evaluation():
    """Test the batch evaluation functionality"""
    
    # Check if required files exist
    checkpoint_path = "checkpoints/unet1d_pit_case1_N3_best.pth"
    data_path = "data/deinterleaving_u_net_case_1_2_scenario_data.npy"
    labels_path = "data/deinterleaving_u_net_case_1_2_scenario_labels.npy"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return False
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return False
        
    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found: {labels_path}")
        return False
    
    try:
        # Load data
        print("Loading data...")
        X = np.load(data_path)
        Y = np.load(labels_path)
        
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {Y.shape}")
        
        # Initialize inference
        print("Initializing inference...")
        inference = RadarDeinterleavingInference(checkpoint_path)
        
        # Set normalization stats
        print("Setting normalization stats...")
        inference.set_normalization_stats(X[:10])
        
        # Test batch evaluation on first 5 frames
        print("Running batch evaluation on first 5 frames...")
        metrics = inference.evaluate_batch(X, Y, start_frame=0, end_frame=5, threshold=0.5)
        
        print("\n" + "="*50)
        print("BATCH EVALUATION TEST RESULTS")
        print("="*50)
        print(f"Frames evaluated: {metrics['num_frames']}")
        print(f"Total assignments: {metrics['num_assignments']}")
        print(f"Average Dice Score: {metrics['dice']:.4f}")
        print(f"Average Precision: {metrics['precision']:.4f}")
        print(f"Average Recall: {metrics['recall']:.4f}")
        print(f"Average F1 Score: {metrics['f1']:.4f}")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing batch evaluation functionality...")
    success = test_batch_evaluation()
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
        sys.exit(1)
