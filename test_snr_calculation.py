#!/usr/bin/env python3
"""
Test script for new SNR calculation method
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import RadarDeinterleavingInference

def test_snr_calculation():
    """Test the new SNR calculation method"""
    
    # Create a simple test case
    print("Testing new SNR calculation method...")
    print("="*50)
    
    # Create test data
    L = 100  # Time samples
    num_emitters = 3
    
    # Create ground truth masks (pulses at specific times)
    gt_masks = np.zeros((num_emitters, L))
    gt_masks[0, 20:30] = 1  # Emitter 0 active from 20-29
    gt_masks[1, 40:50] = 1  # Emitter 1 active from 40-49
    gt_masks[2, 60:70] = 1  # Emitter 2 active from 60-69
    
    # Create ground truth signal (clean pulses)
    gt_signal = np.zeros((2, L))
    gt_signal[0, 20:30] = 2.0  # I channel pulses
    gt_signal[1, 20:30] = 1.5
    gt_signal[0, 40:50] = 1.8
    gt_signal[1, 40:50] = 2.2
    gt_signal[0, 60:70] = 2.5
    gt_signal[1, 60:70] = 1.8
    
    # Create noisy input signal
    noise_level = 0.5
    input_signal = gt_signal + np.random.normal(0, noise_level, gt_signal.shape)
    
    # Calculate noise signal
    noise_signal = input_signal - gt_signal
    
    # Create pulse mask
    pulse_mask = gt_masks.sum(axis=0) > 0
    
    print(f"Pulse regions: {np.sum(pulse_mask)} samples out of {L}")
    print(f"Silence regions: {np.sum(~pulse_mask)} samples out of {L}")
    print()
    
    # Test old method (without pulse mask)
    print("OLD METHOD (all samples):")
    old_snr = calculate_snr_old(gt_signal, noise_signal)
    print(f"SNR: {old_snr:.2f} dB")
    print()
    
    # Test new method (with pulse mask)
    print("NEW METHOD (pulse regions only):")
    new_snr = calculate_snr_new(gt_signal, noise_signal, pulse_mask)
    print(f"SNR: {new_snr:.2f} dB")
    print()
    
    # Show signal power comparison
    print("SIGNAL POWER ANALYSIS:")
    all_signal_power = np.mean(gt_signal**2)
    pulse_signal_power = np.mean(gt_signal[:, pulse_mask]**2)
    silence_signal_power = np.mean(gt_signal[:, ~pulse_mask]**2)
    
    print(f"All samples signal power: {all_signal_power:.4f}")
    print(f"Pulse regions signal power: {pulse_signal_power:.4f}")
    print(f"Silence regions signal power: {silence_signal_power:.4f}")
    print()
    
    print("NOISE POWER ANALYSIS:")
    all_noise_power = np.mean(noise_signal**2)
    pulse_noise_power = np.mean(noise_signal[:, pulse_mask]**2)
    silence_noise_power = np.mean(noise_signal[:, ~pulse_mask]**2)
    
    print(f"All samples noise power: {all_noise_power:.4f}")
    print(f"Pulse regions noise power: {pulse_noise_power:.4f}")
    print(f"Silence regions noise power: {silence_noise_power:.4f}")

def calculate_snr_old(signal_data, noise_data):
    """Old SNR calculation method"""
    signal_power = np.mean(signal_data**2)
    noise_power = np.mean(noise_data**2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    return snr_db

def calculate_snr_new(signal_data, noise_data, pulse_mask):
    """New SNR calculation method"""
    if pulse_mask is not None:
        # Use only pulse regions for signal power
        pulse_signal = signal_data[:, pulse_mask]
        # Use non-pulse regions for noise power
        silence_mask = ~pulse_mask
        silence_noise = noise_data[:, silence_mask]
        
        # Calculate signal power from pulse regions only
        signal_power = np.mean(pulse_signal**2)
        
        # Calculate noise power from silence regions only
        if np.any(silence_mask):
            noise_power = np.mean(silence_noise**2)
        else:
            # If no silence regions, use all noise data
            noise_power = np.mean(noise_data**2)
    else:
        # Fallback to original method if no pulse mask provided
        signal_power = np.mean(signal_data**2)
        noise_power = np.mean(noise_data**2)
    
    # Avoid division by zero
    if noise_power == 0:
        return float('inf')
    
    # Calculate SNR in dB
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db

if __name__ == "__main__":
    test_snr_calculation()
