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
    """
    Apply PIT assignment to match predictions with ground truth
    Returns: matched_predictions, assignment_info
    """
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

def visualize_radar_deinterleaving(frame_idx, X_val, Y_val, matched_predictions, 
                                 assignment_info, save_path=None, show_plots=True, 
                                 local_frame_idx=0):
    """
    Visualize radar deinterleaving results for a specific frame
    """
    if frame_idx >= X_val.shape[0]:
        print(f"Frame index {frame_idx} out of range. Max: {X_val.shape[0]-1}")
        return
    
    # Get data for the frame
    I_data = X_val[frame_idx, 0, :]  # I channel
    Q_data = X_val[frame_idx, 1, :]  # Q channel
    gt_masks = Y_val[frame_idx, :, :]  # Ground truth masks
    pred_masks = matched_predictions[local_frame_idx, :, :]  # Predicted masks (use local index)
    
    # Calculate magnitude from I/Q data
    magnitude = np.sqrt(I_data**2 + Q_data**2)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1 + gt_masks.shape[0], 1, figsize=(15, 6 + 2*gt_masks.shape[0]))
    fig.suptitle(f'Radar Deinterleaving Results - Frame {frame_idx}', fontsize=16, fontweight='bold')
    
    # Plot magnitude signal with I/Q components
    axes[0].plot(magnitude, 'g-', linewidth=1.2, alpha=0.9, label='Magnitude')
    axes[0].plot(I_data, 'b-', linewidth=0.5, alpha=0.6, label='I Channel')
    axes[0].plot(Q_data, 'r-', linewidth=0.5, alpha=0.6, label='Q Channel')
    axes[0].set_title('Radar Signal (Magnitude + I/Q Components)', fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot ground truth and predictions for each emitter
    colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i in range(gt_masks.shape[0]):
        ax_idx = 1 + i
        
        # Ground truth - solid line
        axes[ax_idx].plot(gt_masks[i], color=colors[i % len(colors)], 
                         linewidth=2, alpha=0.8, label=f'Ground Truth')
        
        # Prediction - scatter plot for better visibility
        pred_indices = np.where(pred_masks[i] > 0.5)[0]
        if len(pred_indices) > 0:
            axes[ax_idx].scatter(pred_indices, pred_masks[i][pred_indices], 
                               color='red', s=20, alpha=0.9, marker='o', 
                               label=f'Prediction', zorder=5)
        
        # Also show prediction as line for continuous view
        axes[ax_idx].plot(pred_masks[i], color='red', 
                         linewidth=1, alpha=0.6, linestyle='--', 
                         label=f'Prediction Line')
        
        axes[ax_idx].set_title(f'Emitter {i+1}', fontweight='bold')
        axes[ax_idx].set_ylabel('Activity')
        axes[ax_idx].legend(loc='upper right', fontsize=10)
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].set_ylim(-0.1, 1.1)
    
    axes[-1].set_xlabel('Time Samples')
    
    # Add assignment information
    if local_frame_idx < len(assignment_info):
        info = assignment_info[local_frame_idx]
        info_text = f"Active Emitters: {info['active_emitters']}\n"
        for assign in info['assignments']:
            info_text += f"Pred Ch{assign['pred_channel']} → GT Ch{assign['gt_channel']} (Dice: {assign['dice_score']:.3f})\n"
        
        fig.text(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def test_model_on_validation(model, X_val, Y_val, device='cpu', num_frames=5, start_frame=0):
    """
    Test model on validation data and return results
    Note: X_val and Y_val are already windowed from training
    """
    print(f"Testing model on {num_frames} validation frames...")
    print(f"Input shapes: X_val {X_val.shape}, Y_val {Y_val.shape}")
    
    model.to(device)
    model.eval()
    
    # Normalize validation data (same as training)
    X_val_norm = X_val.copy()
    for ch in range(X_val.shape[1]):
        ch_mean = X_val[:, ch, :].mean()
        ch_std = X_val[:, ch, :].std()
        X_val_norm[:, ch, :] = (X_val[:, ch, :] - ch_mean) / (ch_std + 1e-8)
    
    # Process frames (already windowed)
    end_frame = min(start_frame + num_frames, X_val.shape[0])
    actual_frames = end_frame - start_frame
    all_pred_logits = []
    all_matched_predictions = []
    all_assignment_info = []
    
    for i in range(start_frame, end_frame):
        print(f"  Processing window {i+1-start_frame+1}/{actual_frames} (frame {i})...")
        
        # Get window data (already windowed)
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
        
        # Clear GPU memory after each window
        del X_tensor, Y_tensor, pred_logits, matched_predictions
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Concatenate results
    pred_logits = torch.cat(all_pred_logits, dim=0)
    matched_predictions = torch.cat(all_matched_predictions, dim=0)
    
    # Calculate metrics using validation data
    dice_scores = []
    f1_scores = []
    
    for b in range(actual_frames):
        gt_b = torch.from_numpy(Y_val[start_frame + b]).float()  # Use correct frame index
        pred_b = matched_predictions[b]
        
        # Calculate Dice score for each emitter
        for i in range(gt_b.shape[0]):
            gt_mask = gt_b[i] > 0.5
            pred_mask = pred_b[i] > 0.5
            
            if gt_mask.sum() > 0 or pred_mask.sum() > 0:
                intersection = (gt_mask & pred_mask).sum().float()
                union = gt_mask.sum().float() + pred_mask.sum().float()
                dice = (2 * intersection) / (union + 1e-6)
                dice_scores.append(dice.item())
        
        # Calculate F1 score
        tp = ((gt_b > 0.5) & (pred_b > 0.5)).sum().float()
        fp = ((gt_b <= 0.5) & (pred_b > 0.5)).sum().float()
        fn = ((gt_b > 0.5) & (pred_b <= 0.5)).sum().float()
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        f1_scores.append(f1.item())
    
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    
    print(f"Test Results:")
    print(f"  - Average Dice Score: {avg_dice:.4f}")
    print(f"  - Average F1 Score: {avg_f1:.4f}")
    print(f"  - Frames tested: {actual_frames} (frames {start_frame} to {end_frame-1})")
    
    return pred_logits, matched_predictions, assignment_info, avg_dice, avg_f1

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained radar deinterleaving model')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Path to validation data (.npz file)')
    parser.add_argument('--frames', type=int, default=5,
                       help='Number of frames to visualize')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Starting frame index (0-based)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to files')
    parser.add_argument('--output_dir', type=str, default='test_results',
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
    
    # Load model and validation data
    model, config = load_model_and_config(args.checkpoint)
    X_val, Y_val, val_indices, windowing_config = load_validation_data(args.val_data)
    
    # Test model (validation data is already windowed)
    pred_logits, matched_predictions, assignment_info, avg_dice, avg_f1 = test_model_on_validation(
        model, X_val, Y_val, device, args.frames, start_frame=args.start_frame
    )
    
    # Visualize results
    print(f"\nGenerating visualizations for {args.frames} frames starting from frame {args.start_frame}...")
    for i in range(min(args.frames, X_val.shape[0] - args.start_frame)):
        frame_idx = args.start_frame + i
        local_idx = i  # Local index in matched_predictions
        save_path = os.path.join(args.output_dir, f'frame_{frame_idx:03d}_deinterleaving.png') if args.save_plots else None
        visualize_radar_deinterleaving(
            frame_idx, X_val, Y_val, matched_predictions, assignment_info,
            save_path=save_path, show_plots=not args.save_plots, local_frame_idx=local_idx
        )
    
    print(f"\nTest completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
