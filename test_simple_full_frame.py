#!/usr/bin/env python3
"""
Simple test: Show multiple consecutive windows as one frame
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from modules import UNet1D

def load_model_and_config(checkpoint_path):
    """Load trained model and configuration"""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    
    # Create model
    mcfg = config["model"]
    model = UNet1D(
        in_channels=mcfg["in_channels"],
        num_emitters=mcfg["num_emitters"],
        base_ch=mcfg["base_ch"],
        depth=mcfg["depth"],
        norm=mcfg["norm"],
        dropout=mcfg["dropout"],
        up_mode=mcfg["up_mode"],
        bottleneck_dilation=mcfg["bottleneck_dilation"],
        encoder_kernel_sizes=mcfg["encoder_kernel_sizes"],
        groups_gn=mcfg.get("groups_gn", 8),
        use_residual=mcfg.get("use_residual", False),
    )
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  - Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"  - Val Loss: {checkpoint.get('val_loss', 'Unknown'):.4f}")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config

def load_validation_data(val_data_path):
    """Load validation data saved during training"""
    print(f"Loading validation data from: {val_data_path}")
    
    data = np.load(val_data_path, allow_pickle=True)
    X_val = data['X_val']
    Y_val = data['Y_val']
    val_indices = data['val_indices']
    
    # Load windowing config if available
    windowing_config = None
    if 'windowing_config' in data:
        windowing_config = data['windowing_config'].item()
        print(f"Windowing config loaded: {windowing_config}")
    
    print(f"Validation data loaded:")
    print(f"  - Samples: {X_val.shape[0]}")
    print(f"  - Input shape: {X_val.shape}")
    print(f"  - Labels shape: {Y_val.shape}")
    if windowing_config:
        print(f"  - Total windows: {windowing_config.get('total_windows', 'unknown')}")
        print(f"  - Window length: {windowing_config.get('window_len', 'unknown')}")
        print(f"  - Window stride: {windowing_config.get('window_stride', 'unknown')}")
    
    return X_val, Y_val, val_indices, windowing_config

def apply_pit_assignment(pred_logits, gt_masks, threshold=0.5):
    """Apply PIT assignment to match predictions with ground truth"""
    B, N, T = pred_logits.shape
    probs = torch.sigmoid(pred_logits)
    
    matched_predictions = torch.zeros_like(gt_masks)
    assignment_info = []
    
    for b in range(B):
        pred_b = probs[b]  # (N, T)
        gt_b = gt_masks[b]  # (N, T)
        
        # Find non-empty ground truth channels
        gt_sums = gt_b.sum(dim=-1)
        nonempty_idx = torch.nonzero(gt_sums > 0, as_tuple=False).flatten()
        
        if nonempty_idx.numel() == 0:
            # No active emitters
            assignment_info.append({
                'frame': b,
                'active_emitters': 0,
                'assignments': [],
                'matched_predictions': matched_predictions[b]
            })
            continue
        
        # Create cost matrix using soft dice
        Egt = nonempty_idx.numel()
        P = pred_b.unsqueeze(1).expand(N, Egt, T)
        G = gt_b[nonempty_idx].unsqueeze(0).expand(N, Egt, T)
        inter = 2.0 * (P * G).sum(dim=-1)
        union = P.sum(dim=-1) + G.sum(dim=-1) + 1e-6
        dice_pair = inter / union
        cost = (1.0 - dice_pair).detach().cpu().numpy()
        
        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Apply assignments
        assignments = []
        for i, j_local in zip(row_ind, col_ind):
            j = nonempty_idx[j_local].item()
            matched_predictions[b, j] = (pred_b[i] >= threshold).float()
            assignments.append({
                'pred_channel': int(i),
                'gt_channel': int(j),
                'dice_score': float(dice_pair[i, j_local].item())
            })
        
        assignment_info.append({
            'frame': b,
            'active_emitters': len(assignments),
            'assignments': assignments,
            'matched_predictions': matched_predictions[b]
        })
    
    return matched_predictions, assignment_info

def find_windows_for_frame(frame_idx, windowing_config):
    """Find which windows belong to a specific frame"""
    window_len = windowing_config['window_len']
    window_stride = windowing_config['window_stride']
    
    # Calculate how many windows per frame
    windows_per_frame = (45000 - window_len) // window_stride + 1
    if (45000 - window_len) % window_stride != 0:
        windows_per_frame += 1
    
    # Find the starting window index for this frame
    frame_start_window = frame_idx * windows_per_frame
    
    # Get all windows for this frame
    window_indices = list(range(frame_start_window, frame_start_window + windows_per_frame))
    
    print(f"Frame {frame_idx} uses windows: {window_indices}")
    print(f"Windows per frame: {windows_per_frame}")
    return window_indices

def test_consecutive_windows(model, X_val, Y_val, device='cpu', num_windows=5, start_window=0):
    """Test model on consecutive windows"""
    print(f"Testing model on {num_windows} consecutive windows starting from window {start_window}...")
    
    model.to(device)
    model.eval()
    
    # Normalize validation data
    X_val_norm = X_val.copy()
    for ch in range(X_val.shape[1]):
        ch_mean = X_val[:, ch, :].mean()
        ch_std = X_val[:, ch, :].std()
        X_val_norm[:, ch, :] = (X_val[:, ch, :] - ch_mean) / (ch_std + 1e-8)
    
    # Process windows
    end_window = min(start_window + num_windows, X_val.shape[0])
    actual_windows = end_window - start_window
    all_pred_logits = []
    all_matched_predictions = []
    all_assignment_info = []
    
    for i in range(start_window, end_window):
        print(f"  Processing window {i+1-start_window+1}/{actual_windows} (window {i})...")
        
        # Get window data
        window_X = X_val_norm[i:i+1]  # (1, 2, window_len)
        window_Y = Y_val[i:i+1]       # (1, N, window_len)
        
        print(f"    Window shapes: X {window_X.shape}, Y {window_Y.shape}")
        
        # Convert to torch tensors
        X_tensor = torch.from_numpy(window_X).float().to(device)
        Y_tensor = torch.from_numpy(window_Y).float().to(device)
        
        # Get predictions
        with torch.no_grad():
            pred_logits = model(X_tensor)
        
        print(f"    Prediction shape: {pred_logits.shape}")
        
        # Apply PIT assignment
        matched_predictions, assignment_info = apply_pit_assignment(pred_logits, Y_tensor)
        
        # Store results
        all_pred_logits.append(pred_logits.cpu())
        all_matched_predictions.append(matched_predictions.cpu())
        all_assignment_info.extend(assignment_info)
        
        # Clear GPU memory
        del X_tensor, Y_tensor, pred_logits, matched_predictions
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Concatenate results
    pred_logits = torch.cat(all_pred_logits, dim=0)
    matched_predictions = torch.cat(all_matched_predictions, dim=0)
    
    return pred_logits, matched_predictions, all_assignment_info

def calculate_performance_metrics(matched_predictions, Y_val, start_window, num_windows):
    """Calculate performance metrics for the consecutive windows"""
    dice_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for i in range(num_windows):
        window_idx = start_window + i
        if window_idx >= Y_val.shape[0]:
            break
            
        # Get ground truth and prediction for this window
        gt_window = Y_val[window_idx]  # (N, window_len) - numpy array
        pred_window = matched_predictions[i].numpy()  # (N, window_len) - convert to numpy
        
        # Calculate metrics for each emitter
        for emitter in range(gt_window.shape[0]):
            gt_mask = gt_window[emitter] > 0.5
            pred_mask = pred_window[emitter] > 0.5
            
            if gt_mask.sum() > 0 or pred_mask.sum() > 0:
                # Dice score
                intersection = (gt_mask & pred_mask).sum()
                union = gt_mask.sum() + pred_mask.sum()
                dice = (2 * intersection) / (union + 1e-6)
                dice_scores.append(dice)
                
                # Precision, Recall, F1
                tp = (gt_mask & pred_mask).sum()
                fp = ((~gt_mask) & pred_mask).sum()
                fn = (gt_mask & (~pred_mask)).sum()
                
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
    
    # Calculate averages
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    avg_precision = np.mean(precision_scores) if precision_scores else 0.0
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    
    # Calculate average number of active emitters per window
    active_emitters_per_window = []
    for i in range(num_windows):
        window_idx = start_window + i
        if window_idx >= Y_val.shape[0]:
            break
        gt_window = Y_val[window_idx]  # (N, window_len)
        active_emitters = np.sum(gt_window.sum(axis=1) > 0)  # Count emitters with activity
        active_emitters_per_window.append(active_emitters)
    
    avg_active_emitters = np.mean(active_emitters_per_window) if active_emitters_per_window else 0.0
    
    return {
        'dice': avg_dice,
        'f1': avg_f1,
        'precision': avg_precision,
        'recall': avg_recall,
        'num_emitters': avg_active_emitters
    }

def visualize_consecutive_windows(start_window, X_val, Y_val, matched_predictions, 
                                 assignment_info, windowing_config, performance_metrics=None, save_path=None, show_plots=True):
    """Visualize consecutive windows as one continuous frame"""
    
    window_len = windowing_config['window_len']
    window_stride = windowing_config['window_stride']
    num_windows = len(matched_predictions)
    
    # Calculate total length
    total_len = (num_windows - 1) * window_stride + window_len
    
    # Concatenate windows
    X_concat = np.zeros((2, total_len))
    Y_concat = np.zeros((Y_val.shape[1], total_len))
    pred_concat = np.zeros((Y_val.shape[1], total_len))
    
    for i in range(num_windows):
        start_pos = i * window_stride
        end_pos = start_pos + window_len
        
        # Get window data
        window_X = X_val[start_window + i]  # (2, window_len)
        window_Y = Y_val[start_window + i]  # (N, window_len)
        window_pred = matched_predictions[i]  # (N, window_len)
        
        # Place in concatenated array
        X_concat[:, start_pos:end_pos] = window_X
        Y_concat[:, start_pos:end_pos] = window_Y
        pred_concat[:, start_pos:end_pos] = window_pred.numpy()
    
    # Get I/Q data
    I_data = X_concat[0, :]  # I channel
    Q_data = X_concat[1, :]  # Q channel
    gt_masks = Y_concat  # Ground truth masks
    pred_masks = pred_concat  # Predicted masks
    
    # Calculate magnitude from I/Q data
    magnitude = np.sqrt(I_data**2 + Q_data**2)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1 + gt_masks.shape[0], 1, figsize=(20, 8 + 2*gt_masks.shape[0]))
    fig.suptitle(f'Consecutive Windows as Full Frame - Windows {start_window} to {start_window + num_windows - 1} ({total_len} samples)', 
                 fontsize=16, fontweight='bold')
    
    # Plot magnitude signal
    axes[0].plot(magnitude, 'g-', linewidth=0.8, alpha=0.8, label='Magnitude')
    axes[0].set_title('Radar Signal Magnitude (Concatenated Windows)', fontweight='bold')
    axes[0].set_ylabel('Magnitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add vertical lines to show window boundaries
    for i in range(1, num_windows):
        x_pos = i * window_stride
        axes[0].axvline(x=x_pos, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot ground truth and predictions for each emitter
    colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i in range(gt_masks.shape[0]):
        ax_idx = 1 + i
        
        # Ground truth - solid line
        axes[ax_idx].plot(gt_masks[i], color=colors[i % len(colors)], 
                         linewidth=1, alpha=0.8, label=f'Ground Truth')
        
        # Prediction - scatter plot for better visibility
        pred_indices = np.where(pred_masks[i] > 0.5)[0]
        if len(pred_indices) > 0:
            axes[ax_idx].scatter(pred_indices, pred_masks[i][pred_indices], 
                               color='red', s=8, alpha=0.9, marker='o', 
                               label=f'Prediction', zorder=5)
        
        # Also show prediction as line for continuous view
        axes[ax_idx].plot(pred_masks[i], color='red', 
                         linewidth=0.5, alpha=0.6, linestyle='--', 
                         label=f'Prediction Line')
        
        # Add vertical lines to show window boundaries
        for j in range(1, num_windows):
            x_pos = j * window_stride
            axes[ax_idx].axvline(x=x_pos, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        axes[ax_idx].set_title(f'Emitter {i+1} (Concatenated Windows)', fontweight='bold')
        axes[ax_idx].set_ylabel('Activity')
        axes[ax_idx].legend(loc='upper right', fontsize=10)
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].set_ylim(-0.1, 1.1)
    
    axes[-1].set_xlabel('Time Samples (Concatenated)')
    
    # Add performance metrics or window information
    if performance_metrics:
        info_text = f"Performance Metrics:\n"
        info_text += f"Dice Score: {performance_metrics['dice']:.4f}\n"
        info_text += f"F1 Score: {performance_metrics['f1']:.4f}\n"
        info_text += f"Precision: {performance_metrics['precision']:.4f}\n"
        info_text += f"Recall: {performance_metrics['recall']:.4f}\n"
        info_text += f"Active Emitters: {performance_metrics['num_emitters']}\n"
        info_text += f"Windows: {start_window} to {start_window + num_windows - 1}\n"
        info_text += f"Total samples: {total_len}"
        
        fig.text(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    else:
        info_text = f"Windows: {start_window} to {start_window + num_windows - 1}\n"
        info_text += f"Total samples: {total_len}\n"
        info_text += f"Window length: {window_len}, Stride: {window_stride}\n"
        
        fig.text(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model on consecutive windows as full frame')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Path to validation data (.npz file)')
    parser.add_argument('--windows', type=int, default=5,
                       help='Number of consecutive windows to visualize')
    parser.add_argument('--start_window', type=int, default=0,
                       help='Starting window index (0-based)')
    parser.add_argument('--frame_id', type=int, default=None,
                       help='Frame ID to test (if specified, will find windows for this frame)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to files')
    parser.add_argument('--output_dir', type=str, default='test_results_consecutive',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and data
    model, config = load_model_and_config(args.checkpoint)
    X_val, Y_val, val_indices, windowing_config = load_validation_data(args.val_data)
    
    # Determine window indices
    if args.frame_id is not None:
        # Use frame ID to find windows
        window_indices = find_windows_for_frame(args.frame_id, windowing_config)
        start_window = window_indices[0]
        num_windows = len(window_indices)
        print(f"Frame {args.frame_id} corresponds to windows {start_window} to {start_window + num_windows - 1}")
    else:
        # Use direct window specification
        start_window = args.start_window
        num_windows = args.windows
    
    # Test consecutive windows
    pred_logits, matched_predictions, assignment_info = test_consecutive_windows(
        model, X_val, Y_val, device, num_windows, start_window
    )
    
    # Calculate performance metrics
    print(f"\nCalculating performance metrics...")
    performance_metrics = calculate_performance_metrics(matched_predictions, Y_val, start_window, num_windows)
    print(f"Performance Metrics:")
    print(f"  - Dice Score: {performance_metrics['dice']:.4f}")
    print(f"  - F1 Score: {performance_metrics['f1']:.4f}")
    print(f"  - Precision: {performance_metrics['precision']:.4f}")
    print(f"  - Recall: {performance_metrics['recall']:.4f}")
    print(f"  - Active Emitters: {performance_metrics['num_emitters']}")
    
    # Visualize results
    print(f"\nGenerating visualization for consecutive windows...")
    if args.frame_id is not None:
        save_path = os.path.join(args.output_dir, f'frame_{args.frame_id}_windows_{start_window}_{num_windows}.png') if args.save_plots else None
    else:
        save_path = os.path.join(args.output_dir, f'consecutive_windows_{start_window}_{num_windows}.png') if args.save_plots else None
    
    visualize_consecutive_windows(
        start_window, X_val, Y_val, matched_predictions, assignment_info,
        windowing_config, performance_metrics=performance_metrics, save_path=save_path, show_plots=not args.save_plots
    )
    
    print(f"\nConsecutive windows test completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
