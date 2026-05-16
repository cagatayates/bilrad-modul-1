#!/usr/bin/env python3
"""
Result comparison script for analyzing performance metrics across different datasets.

This script:
1. Loads results from two CSV files (main and denoised)
2. Sorts the main results by SNR (ascending)
3. Extracts frame IDs from a specified index range
4. Finds corresponding metrics in the denoised results
5. Calculates and displays average Dice Score and SNR
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os

def load_csv_results(filepath):
    """
    Load results from CSV file
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        pandas DataFrame with results
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from {filepath}")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def sort_by_snr(df):
    """
    Sort DataFrame by SNR in ascending order
    
    Args:
        df: DataFrame with SNR column
        
    Returns:
        Sorted DataFrame
    """
    if 'snr' not in df.columns:
        print("Error: SNR column not found in data")
        return None
    
    # Sort by SNR (ascending: small to large)
    sorted_df = df.sort_values('snr', ascending=True).reset_index(drop=True)
    print(f"Sorted {len(sorted_df)} records by SNR (ascending)")
    return sorted_df

def extract_frame_ids_by_range(sorted_df, start_idx, end_idx):
    """
    Extract frame IDs from specified index range
    
    Args:
        sorted_df: Sorted DataFrame
        start_idx: Starting index (0-based)
        end_idx: Ending index (exclusive)
        
    Returns:
        List of frame IDs
    """
    if start_idx < 0 or end_idx > len(sorted_df) or start_idx >= end_idx:
        print(f"Error: Invalid index range [{start_idx}, {end_idx})")
        return None
    
    frame_ids = sorted_df.iloc[start_idx:end_idx]['frame_id'].tolist()
    print(f"Extracted frame IDs from index range [{start_idx}, {end_idx}): {len(frame_ids)} frames")
    print(f"Frame IDs: {frame_ids}")
    return frame_ids

def find_corresponding_metrics(denoised_df, frame_ids):
    """
    Find corresponding metrics in denoised results for given frame IDs
    
    Args:
        denoised_df: DataFrame with denoised results
        frame_ids: List of frame IDs to find
        
    Returns:
        DataFrame with corresponding metrics
    """
    # Filter denoised results for the specified frame IDs
    corresponding_df = denoised_df[denoised_df['frame_id'].isin(frame_ids)]
    
    print(f"Found {len(corresponding_df)} corresponding records in denoised results")
    
    if len(corresponding_df) == 0:
        print("Warning: No corresponding records found")
        return None
    
    return corresponding_df

def calculate_averages(df):
    """
    Calculate average Dice Score and SNR
    
    Args:
        df: DataFrame with metrics
        
    Returns:
        Dictionary with average values
    """
    if df is None or len(df) == 0:
        return None
    
    averages = {}
    
    if 'dice_score' in df.columns:
        averages['avg_dice_score'] = df['dice_score'].mean()
        averages['std_dice_score'] = df['dice_score'].std()
        averages['min_dice_score'] = df['dice_score'].min()
        averages['max_dice_score'] = df['dice_score'].max()
    
    if 'snr' in df.columns:
        averages['avg_snr'] = df['snr'].mean()
        averages['std_snr'] = df['snr'].std()
        averages['min_snr'] = df['snr'].min()
        averages['max_snr'] = df['snr'].max()
    
    averages['num_frames'] = len(df)
    
    return averages

def print_results(averages, frame_ids):
    """
    Print comparison results
    
    Args:
        averages: Dictionary with average metrics
        frame_ids: List of frame IDs used
    """
    if averages is None:
        print("No results to display")
        return
    
    print("\n" + "="*60)
    print("RESULT COMPARISON ANALYSIS")
    print("="*60)
    print(f"Number of frames analyzed: {averages['num_frames']}")
    print(f"Frame IDs: {frame_ids}")
    print()
    
    if 'avg_dice_score' in averages:
        print("DICE SCORE STATISTICS:")
        print(f"  Average: {averages['avg_dice_score']:.4f}")
        print(f"  Std Dev: {averages['std_dice_score']:.4f}")
        print(f"  Min:     {averages['min_dice_score']:.4f}")
        print(f"  Max:     {averages['max_dice_score']:.4f}")
        print()
    
    if 'avg_snr' in averages:
        print("SNR STATISTICS:")
        print(f"  Average: {averages['avg_snr']:.2f} dB")
        print(f"  Std Dev: {averages['std_snr']:.2f} dB")
        print(f"  Min:     {averages['min_snr']:.2f} dB")
        print(f"  Max:     {averages['max_snr']:.2f} dB")
        print()
    
    print("="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Compare results between main and denoised datasets')
    parser.add_argument('--main-results', type=str, required=True,
                       help='Path to main results CSV file')
    parser.add_argument('--denoised-results', type=str, required=True,
                       help='Path to denoised results CSV file')
    parser.add_argument('--start-index', type=int, required=True,
                       help='Starting index for frame selection (0-based)')
    parser.add_argument('--end-index', type=int, required=True,
                       help='Ending index for frame selection (exclusive)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.main_results):
        print(f"Error: Main results file not found: {args.main_results}")
        return
    
    if not os.path.exists(args.denoised_results):
        print(f"Error: Denoised results file not found: {args.denoised_results}")
        return
    
    print("Loading results files...")
    
    # Load main results
    main_df = load_csv_results(args.main_results)
    if main_df is None:
        return
    
    # Load denoised results
    denoised_df = load_csv_results(args.denoised_results)
    if denoised_df is None:
        return
    
    print("\nSorting main results by SNR...")
    
    # Sort main results by SNR
    sorted_main_df = sort_by_snr(main_df)
    if sorted_main_df is None:
        return
    
    print(f"\nExtracting frame IDs from index range [{args.start_index}, {args.end_index})...")
    
    # Extract frame IDs from specified range
    frame_ids = extract_frame_ids_by_range(sorted_main_df, args.start_index, args.end_index)
    if frame_ids is None:
        return
    
    print(f"\nFinding corresponding metrics in denoised results...")
    
    # Find corresponding metrics in denoised results
    corresponding_df = find_corresponding_metrics(denoised_df, frame_ids)
    if corresponding_df is None:
        return
    
    print(f"\nCalculating averages...")
    
    # Calculate averages
    averages = calculate_averages(corresponding_df)
    
    # Print results
    print_results(averages, frame_ids)
    
    if args.verbose:
        print("\nDETAILED FRAME DATA:")
        print("-" * 40)
        for _, row in corresponding_df.iterrows():
            print(f"Frame {row['frame_id']}: Dice={row['dice_score']:.4f}, SNR={row['snr']:.2f} dB")

if __name__ == "__main__":
    main()
