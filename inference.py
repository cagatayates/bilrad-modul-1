import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
from scipy.optimize import linear_sum_assignment
from modules import UNet1D
from pri_infer import extract_features_from_iq_data, infer as pri_model_infer

class RadarDeinterleavingInference:
    """Radar deinterleaving inference class for single frame testing"""
    
    def __init__(self, checkpoint_path, device='auto'):
        """
        Initialize inference with trained model
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.device = self._setup_device(device)
        self.model, self.config = self._load_model(checkpoint_path)
        self.normalization_stats = None
        
    def _setup_device(self, device):
        """Setup computation device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, checkpoint_path):
        """Load trained model and configuration"""
        print(f"Loading model from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        state_dict = checkpoint['state_dict']

        # Backward-compatible attention selection based on checkpoint keys
        # - Older checkpoints used ChannelAttention1D with keys: bottleneck_attention.fc.*
        # - Newer checkpoints use StatefulChannelAttention1D with keys: bottleneck_attention.fc1/gru_cell/fc2.*
        attention_type = "stateful"
        for k in state_dict.keys():
            if k.startswith("bottleneck_attention.fc."):
                attention_type = "classic"
                break
        
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
            attention_type=attention_type,
            return_state=False,
        )
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        return model, config
    
    def set_normalization_stats(self, X_train):
        """
        Set normalization statistics from training data
        
        Args:
            X_train: Training data array (B, 2, L)
        """
        self.normalization_stats = {}
        for ch in range(X_train.shape[1]):
            self.normalization_stats[f'ch_{ch}_mean'] = X_train[:, ch, :].mean()
            self.normalization_stats[f'ch_{ch}_std'] = X_train[:, ch, :].std()
        print("Normalization statistics set from training data")
    
    def normalize_data(self, X):
        """
        Normalize input data using training statistics
        
        Args:
            X: Input data (2, L) or (B, 2, L)
            
        Returns:
            Normalized data
        """
        if self.normalization_stats is None:
            print("Warning: Normalization stats not set. Using data statistics.")
            X_norm = X.copy()
            for ch in range(X.shape[-2]):
                ch_mean = X[..., ch, :].mean()
                ch_std = X[..., ch, :].std()
                X_norm[..., ch, :] = (X[..., ch, :] - ch_mean) / (ch_std + 1e-8)
            return X_norm
        
        X_norm = X.copy()
        for ch in range(X.shape[-2]):
            mean_key = f'ch_{ch}_mean'
            std_key = f'ch_{ch}_std'
            if mean_key in self.normalization_stats:
                X_norm[..., ch, :] = (X[..., ch, :] - self.normalization_stats[mean_key]) / (self.normalization_stats[std_key] + 1e-8)
        return X_norm
    
    def predict(self, I_data, Q_data, threshold=0.5, use_windowing=True):
        """
        Predict emitter masks for given I/Q data
        
        Args:
            I_data: I channel data (L,)
            Q_data: Q channel data (L,)
            threshold: Threshold for binary mask conversion
            use_windowing: Whether to use windowing (if model was trained with windows)
            
        Returns:
            dict with predictions and metadata
        """
        # Prepare input
        if len(I_data.shape) == 1:
            X = np.stack([I_data, Q_data], axis=0)  # (2, L)
            X = X[np.newaxis, ...]  # (1, 2, L)
        else:
            X = np.stack([I_data, Q_data], axis=-2)  # (..., 2, L)
        
        # Check if model was trained with windowing
        if use_windowing and 'windowing' in self.config:
            windowing_config = self.config['windowing']
            if windowing_config.get('use_windows', False):
                return self._predict_with_windowing(X, threshold, windowing_config)
        
        # Normalize
        X_norm = self.normalize_data(X)
        
        # Convert to tensor
        X_tensor = torch.from_numpy(X_norm).float().to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(X_tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = torch.sigmoid(logits)
            binary_masks = (probs >= threshold).float()
        
        # Convert back to numpy
        logits_np = logits.cpu().numpy()
        probs_np = probs.cpu().numpy()
        masks_np = binary_masks.cpu().numpy()
        
        # Remove batch dimension if single frame
        if len(I_data.shape) == 1:
            logits_np = logits_np[0]
            probs_np = probs_np[0]
            masks_np = masks_np[0]
        
        return {
            'logits': logits_np,
            'probabilities': probs_np,
            'binary_masks': masks_np,
            'threshold': threshold,
            'num_emitters': self.config['model']['num_emitters']
        }
    
    def _predict_with_windowing(self, X, threshold, windowing_config):
        """
        Predict using windowing approach (same as training)
        
        Args:
            X: Input data (B, 2, L)
            threshold: Threshold for binary mask conversion
            windowing_config: Windowing configuration from training
            
        Returns:
            dict with predictions and metadata
        """
        window_len = windowing_config['window_len']
        window_stride = windowing_config['window_stride']
        include_tail_window = windowing_config.get('include_tail_window', True)
        
        B, C, L = X.shape
        num_emitters = self.config['model']['num_emitters']
        
        # Create windows
        windows = []
        window_starts = []
        
        for b in range(B):
            start = 0
            while start + window_len <= L:
                windows.append(X[b, :, start:start + window_len])
                window_starts.append((b, start))
                start += window_stride
            
            # Add tail window if needed
            if include_tail_window and (len(windows) == 0 or window_starts[-1][0] != b or window_starts[-1][1] + window_len < L):
                if L - window_len >= 0:
                    windows.append(X[b, :, L - window_len:L])
                    window_starts.append((b, L - window_len))
        
        if len(windows) == 0:
            # Fallback: use the whole sequence if no windows can be created
            windows = [X[b, :, :] for b in range(B)]
            window_starts = [(b, 0) for b in range(B)]
        
        # Stack windows
        X_windows = np.stack(windows, axis=0)  # (num_windows, 2, window_len)
        
        # Normalize
        X_norm = self.normalize_data(X_windows)
        
        # Convert to tensor
        X_tensor = torch.from_numpy(X_norm).float().to(self.device)
        
        # Predict on windows
        with torch.no_grad():
            logits = self.model(X_tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = torch.sigmoid(logits)
            binary_masks = (probs >= threshold).float()
        
        # Convert back to numpy
        logits_np = logits.cpu().numpy()
        probs_np = probs.cpu().numpy()
        masks_np = binary_masks.cpu().numpy()
        
        # Reconstruct full sequence predictions
        full_logits = np.zeros((B, num_emitters, L))
        full_probs = np.zeros((B, num_emitters, L))
        full_masks = np.zeros((B, num_emitters, L))
        window_counts = np.zeros((B, L))
        
        for i, (b, start) in enumerate(window_starts):
            end = start + window_len
            full_logits[b, :, start:end] += logits_np[i]
            full_probs[b, :, start:end] += probs_np[i]
            full_masks[b, :, start:end] += masks_np[i]
            window_counts[b, start:end] += 1
        
        # Average overlapping regions
        for b in range(B):
            for t in range(L):
                if window_counts[b, t] > 0:
                    full_logits[b, :, t] /= window_counts[b, t]
                    full_probs[b, :, t] /= window_counts[b, t]
                    full_masks[b, :, t] = (full_masks[b, :, t] / window_counts[b, t] >= 0.5).astype(float)
        
        # Remove batch dimension if single frame
        if B == 1:
            full_logits = full_logits[0]
            full_probs = full_probs[0]
            full_masks = full_masks[0]
        
        return {
            'logits': full_logits,
            'probabilities': full_probs,
            'binary_masks': full_masks,
            'threshold': threshold,
            'num_emitters': num_emitters,
            'windowing_info': {
                'num_windows': len(windows),
                'window_len': window_len,
                'window_stride': window_stride,
                'used_windowing': True
            }
        }
    
    def predict_with_pit(self, I_data, Q_data, gt_masks=None, threshold=0.5, use_windowing=True):
        """
        Predict with PIT assignment if ground truth is available
        
        Args:
            I_data: I channel data (L,)
            Q_data: Q channel data (L,)
            gt_masks: Ground truth masks (N, L) - optional
            threshold: Threshold for binary mask conversion
            use_windowing: Whether to use windowing (if model was trained with windows)
            
        Returns:
            dict with predictions and PIT assignment info
        """
        # Get basic predictions
        pred_result = self.predict(I_data, Q_data, threshold, use_windowing)
        
        if gt_masks is None:
            return pred_result
        
        # Apply PIT assignment
        pred_logits = torch.from_numpy(pred_result['logits']).float()
        # Ensure GT is writable / contiguous for torch
        gt_np = np.asarray(gt_masks).copy()
        gt_tensor = torch.from_numpy(gt_np).float()
        
        if len(pred_logits.shape) == 2:
            pred_logits = pred_logits.unsqueeze(0)
            gt_tensor = gt_tensor.unsqueeze(0)
            single_frame = True
        else:
            single_frame = False

        # Align channel count (N) and time length (T) between prediction and GT
        # - If GT has fewer channels than model output, pad GT with zeros.
        # - If GT has more channels, crop extra channels.
        # - If lengths differ, crop both to min length (model may output a slightly different length due to pooling/upsampling).
        _, N_pred, T_pred = pred_logits.shape
        _, N_gt, T_gt = gt_tensor.shape

        if T_gt != T_pred:
            T = min(T_gt, T_pred)
            pred_logits = pred_logits[..., :T]
            gt_tensor = gt_tensor[..., :T]

        if N_gt < N_pred:
            pad = torch.zeros((gt_tensor.shape[0], N_pred - N_gt, gt_tensor.shape[2]), dtype=gt_tensor.dtype, device=gt_tensor.device)
            gt_tensor = torch.cat([gt_tensor, pad], dim=1)
        elif N_gt > N_pred:
            gt_tensor = gt_tensor[:, :N_pred, :]
        
        matched_predictions, assignment_info = self._apply_pit_assignment(pred_logits, gt_tensor)
        
        if single_frame:
            matched_predictions = matched_predictions[0]
            assignment_info = assignment_info[0]
        
        pred_result.update({
            'matched_predictions': matched_predictions.numpy(),
            'assignment_info': assignment_info
        })
        
        return pred_result
    
    def _apply_pit_assignment(self, pred_logits, gt_masks):
        """Apply PIT assignment using Hungarian algorithm"""
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
                assignment_info.append({
                    'active_emitters': 0,
                    'assignments': [],
                    'dice_scores': [],
                    'f1_scores': []
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
            
            # Apply assignments and calculate F1-scores
            assignments = []
            dice_scores = []
            f1_scores = []
            for i, j_local in zip(row_ind, col_ind):
                j = nonempty_idx[j_local].item()
                matched_predictions[b, j] = pred_b[i]
                dice_score = dice_pair[i, j_local].item()
                
                # Calculate F1-score for this assignment
                pred_binary = (pred_b[i] >= 0.5).float()
                gt_binary = gt_b[j]
                
                tp = (pred_binary * gt_binary).sum().item()
                fp = (pred_binary * (1 - gt_binary)).sum().item()
                fn = ((1 - pred_binary) * gt_binary).sum().item()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
                
                assignments.append({
                    'pred_channel': int(i),
                    'gt_channel': int(j),
                    'dice_score': float(dice_score),
                    'f1_score': float(f1_score)
                })
                dice_scores.append(float(dice_score))
                f1_scores.append(float(f1_score))
            
            assignment_info.append({
                'active_emitters': len(assignments),
                'assignments': assignments,
                'dice_scores': dice_scores,
                'f1_scores': f1_scores
            })
        
        return matched_predictions, assignment_info
    
    def calculate_snr(self, signal_data, noise_data, pulse_mask=None):
        """
        Calculate Signal-to-Noise Ratio (SNR) in dB
        
        Args:
            signal_data: Signal data (2, L) - I/Q channels
            noise_data: Noise data (2, L) - I/Q channels
            pulse_mask: Boolean mask indicating pulse regions (L,)
            
        Returns:
            SNR in dB
        """
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

    def evaluate_batch(self, X_data, Y_data, start_frame=0, end_frame=None, threshold=0.5, use_windowing=True, enable_snr=False, gt_data=None, save_csv=None):
        """
        Evaluate model performance on a batch of frames
        
        Args:
            X_data: Input data (B, 2, L)
            Y_data: Ground truth labels (B, N, L)
            start_frame: Starting frame index
            end_frame: Ending frame index (if None, use all frames)
            threshold: Threshold for binary mask conversion
            use_windowing: Whether to use windowing (if model was trained with windows)
            enable_snr: Whether to calculate SNR
            gt_data: Ground truth data for SNR calculation (B, 2, L)
            save_csv: Path to save CSV file with frame-by-frame results
            enable_pri: Whether to run PRI analysis per emitter/frame
            pri_model_ckpt: Path to PRI model checkpoint (.pt)
            pri_gt_json: Path to PRI ground truth JSON (emitter-based)
            pri_fs_hz: Sampling frequency for PRI analysis (Hz)
            
        Returns:
            dict with average performance metrics
        """
        if end_frame is None:
            end_frame = X_data.shape[0]
        
        end_frame = min(end_frame, X_data.shape[0])
        
        all_dice_scores = []
        all_precision_scores = []
        all_recall_scores = []
        all_f1_scores = []
        all_snr_scores = []
        all_pri_mae = []
        all_pri_cls_acc = []
        
        # CSV data collection
        csv_data = []
        
        print(f"Evaluating frames {start_frame} to {end_frame-1}...")
        
        for frame_idx in range(start_frame, end_frame):
            # Get I/Q data for the frame
            I_data = X_data[frame_idx, 0, :]
            Q_data = X_data[frame_idx, 1, :]
            gt_masks = Y_data[frame_idx]
            
            # Calculate SNR if enabled
            frame_snr = None
            if enable_snr and gt_data is not None:
                # Get ground truth signal for this frame
                gt_signal = gt_data[frame_idx]  # (2, L)
                # Calculate noise as difference between input and ground truth
                input_signal = np.stack([I_data, Q_data], axis=0)  # (2, L)
                noise_signal = input_signal - gt_signal
                
                # Create pulse mask: any emitter active at this time
                pulse_mask = gt_masks.sum(axis=0) > 0  # (L,)
                
                frame_snr = self.calculate_snr(gt_signal, noise_signal, pulse_mask)
                all_snr_scores.append(frame_snr)
            
            # Predict with PIT assignment (needed for PRI + metrics)
            result = self.predict_with_pit(I_data, Q_data, gt_masks, threshold, use_windowing)

            # PRI analysis per frame (optional)
            frame_pri_mae = None
            frame_pri_cls_acc = None
            if hasattr(self, "config"):
                pri_cfg = self.config.get("pri", {})
            else:
                pri_cfg = {}

            enable_pri = pri_cfg.get("enable_pri", False)
            pri_model_ckpt = pri_cfg.get("pri_model_ckpt", None)
            pri_gt_json = pri_cfg.get("pri_gt_json", None)
            pri_fs_hz = pri_cfg.get("pri_fs_hz", 3_000_000.0)

            if enable_pri and pri_model_ckpt is not None and pri_gt_json is not None:
                try:
                    pri_result = run_pri_analysis(
                        I_data,
                        Q_data,
                        result,
                        pri_model_ckpt=pri_model_ckpt,
                        fs_hz=pri_fs_hz,
                        gt_json_path=pri_gt_json,
                        frame_idx=frame_idx,
                        csv_prefix=None,
                    )
                    if pri_result is not None:
                        fs = pri_result.get("frame_summary", {})
                        frame_pri_mae = fs.get("pri_mae_us")
                        frame_pri_cls_acc = fs.get("pri_cls_acc")
                        if frame_pri_mae is not None:
                            all_pri_mae.append(frame_pri_mae)
                        if frame_pri_cls_acc is not None:
                            all_pri_cls_acc.append(frame_pri_cls_acc)
                except Exception as e:
                    print(f"PRI analysis failed for frame {frame_idx}: {e}")
            
            # Calculate frame-level metrics
            frame_dice_scores = []
            frame_precision_scores = []
            frame_recall_scores = []
            frame_f1_scores = []
            
            # Extract metrics from assignment info
            if 'assignment_info' in result:
                info = result['assignment_info']
                for assign in info['assignments']:
                    dice_score = assign['dice_score']
                    f1_score = assign['f1_score']
                    
                    frame_dice_scores.append(dice_score)
                    frame_f1_scores.append(f1_score)
                    
                    # Calculate precision and recall
                    pred_channel = assign['pred_channel']
                    gt_channel = assign['gt_channel']
                    
                    pred_binary = (result['probabilities'][pred_channel] >= threshold).astype(float)
                    gt_binary = gt_masks[gt_channel]
                    
                    tp = np.sum(pred_binary * gt_binary)
                    fp = np.sum(pred_binary * (1 - gt_binary))
                    fn = np.sum((1 - pred_binary) * gt_binary)
                    
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    
                    frame_precision_scores.append(precision)
                    frame_recall_scores.append(recall)
                    
                    # Add to global lists
                    all_dice_scores.append(dice_score)
                    all_f1_scores.append(f1_score)
                    all_precision_scores.append(precision)
                    all_recall_scores.append(recall)
            
            # Calculate average metrics for this frame
            avg_frame_dice = np.mean(frame_dice_scores) if frame_dice_scores else 0.0
            avg_frame_precision = np.mean(frame_precision_scores) if frame_precision_scores else 0.0
            avg_frame_recall = np.mean(frame_recall_scores) if frame_recall_scores else 0.0
            avg_frame_f1 = np.mean(frame_f1_scores) if frame_f1_scores else 0.0
            
            # Print frame result with SNR if available
            if frame_snr is not None:
                print(f"Processing frame {frame_idx}, dice score: {avg_frame_dice:.4f}, SNR: {frame_snr:.2f} dB")
            else:
                print(f"Processing frame {frame_idx}, dice score: {avg_frame_dice:.4f}")
            
            # Collect data for CSV export
            csv_row = {
                'frame_id': frame_idx,
                'dice_score': avg_frame_dice,
                'precision': avg_frame_precision,
                'recall': avg_frame_recall,
                'f1_score': avg_frame_f1,
                'snr': frame_snr if frame_snr is not None else None,
                'pri_cls_acc': frame_pri_cls_acc,
                'pri_mae_us': frame_pri_mae,
            }
            csv_data.append(csv_row)
        
        # Calculate average metrics
        avg_dice = np.mean(all_dice_scores) if all_dice_scores else 0.0
        avg_precision = np.mean(all_precision_scores) if all_precision_scores else 0.0
        avg_recall = np.mean(all_recall_scores) if all_recall_scores else 0.0
        avg_f1 = np.mean(all_f1_scores) if all_f1_scores else 0.0
        avg_snr = np.mean(all_snr_scores) if all_snr_scores else None
        avg_pri_mae = np.mean(all_pri_mae) if all_pri_mae else None
        avg_pri_cls_acc = np.mean(all_pri_cls_acc) if all_pri_cls_acc else None
        
        result = {
            'dice': avg_dice,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'num_frames': end_frame - start_frame,
            'num_assignments': len(all_dice_scores)
        }
        
        if avg_snr is not None:
            result['snr'] = avg_snr
        if avg_pri_mae is not None:
            result['pri_mae_us'] = avg_pri_mae
        if avg_pri_cls_acc is not None:
            result['pri_cls_acc'] = avg_pri_cls_acc
        
        # Save CSV file if requested
        if save_csv is not None:
            self._save_csv(csv_data, save_csv, enable_snr)
            
        return result

    def _save_csv(self, csv_data, filepath, enable_snr):
        """
        Save frame-by-frame results to CSV file
        
        Args:
            csv_data: List of dictionaries with frame data
            filepath: Path to save CSV file
            enable_snr: Whether SNR data is available
        """
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                # Define fieldnames based on available data
                fieldnames = [
                    'frame_id',
                    'dice_score',
                    'precision',
                    'recall',
                    'f1_score',
                    'snr',
                    'pri_cls_acc',
                    'pri_mae_us',
                ]
                # If SNR is disabled, we will still keep the column but fill with None
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in csv_data:
                    # Only include SNR if it's available
                    if not enable_snr and 'snr' in row:
                        row_copy = row.copy()
                        del row_copy['snr']
                        writer.writerow(row_copy)
                    else:
                        writer.writerow(row)
            
            print(f"CSV file saved to: {filepath}")
            
        except Exception as e:
            print(f"Error saving CSV file: {e}")

    def visualize_prediction(self, I_data, Q_data, prediction_result, gt_masks=None, 
                           save_path=None, show_plot=True, f1_threshold=0.0):
        """
        Visualize prediction results
        
        Args:
            I_data: I channel data (L,)
            Q_data: Q channel data (L,)
            prediction_result: Result from predict() or predict_with_pit()
            gt_masks: Ground truth masks (N, L) - optional
            save_path: Path to save plot - optional
            show_plot: Whether to display plot
            f1_threshold: Minimum F1 score threshold for plotting emitters (default: 0.0, plots all)
        """
        num_emitters = prediction_result['num_emitters']
        probs = prediction_result['probabilities']
        binary_masks = prediction_result['binary_masks']
        
        # Check if PIT assignment was used
        use_matched_predictions = 'matched_predictions' in prediction_result and gt_masks is not None
        
        # Calculate magnitude from I/Q data
        magnitude = np.sqrt(I_data**2 + Q_data**2)
        
        # Create mapping for PIT assignment if available
        assignment_mapping = {}
        active_emitters = []
        if use_matched_predictions and 'assignment_info' in prediction_result:
            info = prediction_result['assignment_info']
            for assign in info['assignments']:
                # Filter by F1 score threshold
                f1_score = assign.get('f1_score', 0.0)
                if f1_score >= f1_threshold:
                    assignment_mapping[assign['gt_channel']] = assign['pred_channel']
                    active_emitters.append(assign['gt_channel'])
        else:
            # If no PIT assignment, show all emitters (no F1 filtering possible)
            active_emitters = list(range(num_emitters))
        
        # Create figure with only active emitters
        num_plots = 1 + len(active_emitters)  # +1 for magnitude plot
        fig, axes = plt.subplots(num_plots, 1, figsize=(15, 6 + 2*len(active_emitters)))
        if num_plots == 1:
            axes = [axes]  # Make it iterable for single plot
        fig.suptitle('Radar Deinterleaving Prediction', fontsize=16, fontweight='bold')
        
        # Plot magnitude signal with I/Q components
        axes[0].plot(magnitude, 'g-', linewidth=1.2, alpha=0.9, label='Magnitude')
        axes[0].plot(I_data, 'b-', linewidth=0.5, alpha=0.6, label='I Channel')
        axes[0].plot(Q_data, 'r-', linewidth=0.5, alpha=0.6, label='Q Channel')
        axes[0].set_title('Radar Signal (Magnitude + I/Q Components)', fontweight='bold')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot emitter predictions - only active ones
        colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for plot_idx, gt_idx in enumerate(active_emitters):
            ax_idx = 1 + plot_idx
            
            # Determine which prediction to show for this ground truth channel
            if use_matched_predictions and gt_idx in assignment_mapping:
                # Use matched prediction
                pred_idx = assignment_mapping[gt_idx]
                pred_probs = prediction_result['matched_predictions'][gt_idx]
                pred_masks = (pred_probs >= prediction_result['threshold']).astype(float)
                pred_label = f'Pred Ch{pred_idx} → GT Ch{gt_idx}'
                gt_label = f'GT Emitter {gt_idx+1}'
                title = f'GT Emitter {gt_idx+1} (Matched with Pred Ch{pred_idx})'
            else:
                # Use original prediction
                pred_probs = probs[gt_idx]
                pred_masks = binary_masks[gt_idx]
                pred_label = f'Emitter {gt_idx+1} Prob'
                gt_label = f'GT Emitter {gt_idx+1}'
                title = f'Emitter {gt_idx+1} - Prediction'
            
            # Plot probability and binary mask
            axes[ax_idx].plot(pred_probs, color=colors[plot_idx % len(colors)], 
                             linewidth=1.5, alpha=0.8, label=pred_label)
            axes[ax_idx].plot(pred_masks, color=colors[plot_idx % len(colors)], 
                             linewidth=2, alpha=0.6, linestyle='--', label=f'{pred_label} Mask')
            
            # Plot ground truth if available
            if gt_masks is not None:
                axes[ax_idx].plot(gt_masks[gt_idx], color='red', linewidth=1, 
                                 alpha=0.5, linestyle=':', label=gt_label)
            
            # Highlight active regions
            active_indices = np.where(pred_masks > 0.5)[0]
            if len(active_indices) > 0:
                axes[ax_idx].fill_between(active_indices, 0, 1, alpha=0.2, 
                                        color=colors[plot_idx % len(colors)])
            
            axes[ax_idx].set_title(title, fontweight='bold')
            axes[ax_idx].set_ylabel('Activity')
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
            axes[ax_idx].set_ylim(-0.1, 1.1)
        
        axes[-1].set_xlabel('Time Samples')
        
        # Add PIT assignment info if available
        if 'assignment_info' in prediction_result:
            info = prediction_result['assignment_info']
            # Filter assignments by F1 threshold
            filtered_assignments = [a for a in info['assignments'] 
                                  if a.get('f1_score', 0.0) >= f1_threshold]
            info_text = f"Active Emitters (F1 >= {f1_threshold:.3f}): {len(filtered_assignments)}\n"
            for assign in filtered_assignments:
                info_text += f"Pred Ch{assign['pred_channel']} → GT Ch{assign['gt_channel']} (Dice: {assign['dice_score']:.3f}, F1: {assign['f1_score']:.3f})\n"
            
            if len(filtered_assignments) > 0:
                fig.text(0.02, 0.02, info_text, fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            elif f1_threshold > 0.0:
                # Show message if threshold filtered out all emitters
                info_text = f"No emitters with F1 >= {f1_threshold:.3f}"
                fig.text(0.02, 0.02, info_text, fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()


def run_pri_analysis(
    I_data,
    Q_data,
    prediction_result,
    pri_model_ckpt,
    fs_hz: float = 3_000_000.0,
    gt_json_path: str | None = None,
    frame_idx: int | None = None,
    csv_prefix: str | None = None,
    min_active_samples: int = 10,
):
    """
    Run PRI mode classification + PRI regression per emitter using PRI model.

    Args:
        I_data: I channel data (L,)
        Q_data: Q channel data (L,)
        prediction_result: dict from predict() or predict_with_pit()
        pri_model_ckpt: Path to PRI model checkpoint (.pt)
        fs_hz: Sampling frequency (Hz), default 3_000_000
        gt_json_path: Optional path to PRI ground truth JSON.
            Expected to be emitter-based (same GT for all frames), e.g.:
                [
                  {"emitter_id": 0, "mode": "staggered", "pri_us": 1200.0},
                  {"emitter_id": 1, "mode": "jittered",  "pri_us": 800.0}
                ]
        frame_idx: Optional frame index (only used for CSV / logging)
        csv_prefix: Optional prefix for CSV files. Will create
                    `<prefix>_emitters.csv` and `<prefix>_frames.csv`
        min_active_samples: Minimum number of active samples to consider emitter
    """
    num_emitters = prediction_result.get("num_emitters")
    if num_emitters is None:
        print("PRI analysis skipped: 'num_emitters' not found in prediction_result.")
        return None

    threshold = prediction_result.get("threshold", 0.5)

    # If PIT was applied and matched_predictions are available, use them as emitter-wise probs
    if "matched_predictions" in prediction_result:
        probs_emitters = prediction_result["matched_predictions"]
        binary_masks_emitters = (probs_emitters >= threshold).astype(float)
    else:
        probs_emitters = prediction_result["probabilities"]
        binary_masks_emitters = prediction_result["binary_masks"]

    # Safety: number of channels in masks may be different from config['num_emitters']
    num_channels = int(binary_masks_emitters.shape[0])

    I_data = np.asarray(I_data)
    Q_data = np.asarray(Q_data)

    feat_vectors = []
    emitter_ids = []
    emitter_stats = []

    # Collect features per emitter (loop over actually available channels)
    for em_idx in range(num_channels):
        mask = binary_masks_emitters[em_idx]
        active_samples = int(mask.sum())
        if active_samples < min_active_samples:
            continue

        indices = np.where(mask > 0.5)[0]
        if indices.size == 0:
            continue

        iq_em = np.stack([I_data[indices], Q_data[indices]], axis=1)  # (N, 2)

        try:
            features = extract_features_from_iq_data(iq_em, fs_hz=fs_hz)
        except Exception as e:
            print(f"PRI feature extraction failed for emitter {em_idx}: {e}")
            continue

        feat_vec = features.get("feat_vector")
        if feat_vec is None:
            print(f"'feat_vector' not found for emitter {em_idx}, skipping.")
            continue

        pulses = int(features.get("stats", {}).get("pulses", 0))

        feat_vectors.append(feat_vec)
        emitter_ids.append(em_idx)
        emitter_stats.append(
            {
                "active_samples": active_samples,
                "pulses": pulses,
            }
        )

    if not feat_vectors:
        print("No active emitters found for PRI analysis (after masking).")
        return None

    X_feat = np.stack(feat_vectors, axis=0).astype(np.float32)

    # Run PRI model inference
    pri_res = pri_model_infer(pri_model_ckpt, X_feat)

    # Load GT JSON if provided (emitter-based, same GT for all frames)
    gt_by_emitter = {}
    if gt_json_path is not None:
        try:
            with open(gt_json_path, "r", encoding="utf-8") as f:
                pri_gt = json.load(f)

            if isinstance(pri_gt, list):
                items = pri_gt
            elif isinstance(pri_gt, dict):
                # Support {"emitters": [...]} or { "0": {...}, "1": {...} }
                if "emitters" in pri_gt and isinstance(pri_gt["emitters"], list):
                    items = pri_gt["emitters"]
                else:
                    items = list(pri_gt.values())
            else:
                items = []

            for item in items:
                if not isinstance(item, dict):
                    continue
                em_id = int(item.get("emitter_id", -1))
                if em_id >= 0:
                    gt_by_emitter[em_id] = item
        except Exception as e:
            print(f"Warning: could not read PRI GT JSON: {e}")

    # Build per-emitter results
    pri_errors = []
    pri_cls_total = 0
    pri_cls_correct = 0
    per_emitter_rows = []

    for i, em_idx in enumerate(emitter_ids):
        stats = emitter_stats[i]
        probs = pri_res["probs"][i]
        pred_idx = int(pri_res["pred_idx"][i])
        pred_name = str(pri_res["pred_name"][i])
        top_prob = float(probs[pred_idx])

        pri_us = None
        if "pri_us" in pri_res:
            # pri_us can be scalar or array
            pri_val = pri_res["pri_us"][i]
            pri_us = float(pri_val if np.isscalar(pri_val) else pri_val.item())

        gt = gt_by_emitter.get(em_idx)
        if gt is not None:
            gt_mode = gt.get("mode")
            gt_pri_us = gt.get("pri_us")
            if pri_us is not None and gt_pri_us is not None:
                abs_err = float(abs(pri_us - gt_pri_us))
                rel_err = float(abs_err / max(float(gt_pri_us), 1e-6) * 100.0)
                pri_errors.append(abs_err)
            else:
                abs_err = None
                rel_err = None

            # classification correctness (mode)
            if gt_mode is not None:
                pri_cls_total += 1
                if str(gt_mode) == pred_name:
                    pri_cls_correct += 1
        else:
            gt_mode = None
            gt_pri_us = None
            abs_err = None
            rel_err = None

        row = {
            "frame_id": int(frame_idx) if frame_idx is not None else -1,
            "emitter_id": int(em_idx),
            "active_samples": int(stats["active_samples"]),
            "pulses": int(stats["pulses"]),
            "pred_mode": pred_name,
            "pred_mode_idx": pred_idx,
            "pred_pri_us": pri_us,
            "gt_mode": gt_mode,
            "gt_pri_us": gt_pri_us,
            "pri_abs_err_us": abs_err,
            "pri_rel_err_pct": rel_err,
            "top_prob": top_prob,
        }
        per_emitter_rows.append(row)

    # Frame-level summary
    frame_mae = float(np.mean(pri_errors)) if pri_errors else None
    frame_cls_acc = (
        float(pri_cls_correct) / float(pri_cls_total) if pri_cls_total > 0 else None
    )
    frame_summary = {
        "frame_id": int(frame_idx) if frame_idx is not None else -1,
        "num_emitters_total": int(num_emitters),
        "num_emitters_analyzed": int(len(per_emitter_rows)),
        "pri_mae_us": frame_mae,
        "pri_cls_acc": frame_cls_acc,
    }

    # Console output
    print("\n=== PRI ANALYSIS PER EMITTER ===")
    for r in per_emitter_rows:
        msg = (
            f"Emitter {r['emitter_id']}: "
            f"mode_pred={r['pred_mode']}"
        )
        if r["pred_pri_us"] is not None:
            msg += f", pri_pred={r['pred_pri_us']:.2f} us"
        if r["gt_mode"] is not None or r["gt_pri_us"] is not None:
            msg += f" | GT mode={r['gt_mode']}, GT pri={r['gt_pri_us']} us"
        if r["pri_abs_err_us"] is not None:
            msg += f", abs_err={r['pri_abs_err_us']:.2f} us"
        if r["pri_rel_err_pct"] is not None:
            msg += f" ({r['pri_rel_err_pct']:.2f}%)"
        print(msg)

    print("\n=== PRI FRAME SUMMARY ===")
    print(
        f"Frame {frame_summary['frame_id']}: "
        f"emitters_analyzed={frame_summary['num_emitters_analyzed']} / {frame_summary['num_emitters_total']}"
    )
    if frame_summary["pri_mae_us"] is not None:
        print(f"Mean |pred_pri - gt_pri| = {frame_summary['pri_mae_us']:.2f} us")
    if frame_summary["pri_cls_acc"] is not None:
        print(f"PRI mode classification accuracy = {frame_summary['pri_cls_acc']:.3f}")

    # CSV export
    if csv_prefix is not None:
        emit_path = f"{csv_prefix}_emitters.csv"
        frame_path = f"{csv_prefix}_frames.csv"

        # Per-emitter CSV
        emit_fields = [
            "frame_id",
            "emitter_id",
            "active_samples",
            "pulses",
            "pred_mode",
            "pred_mode_idx",
            "pred_pri_us",
            "gt_mode",
            "gt_pri_us",
            "pri_abs_err_us",
            "pri_rel_err_pct",
            "top_prob",
        ]
        try:
            write_header = not os.path.exists(emit_path)
            with open(emit_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=emit_fields)
                if write_header:
                    writer.writeheader()
                for r in per_emitter_rows:
                    writer.writerow(r)
            print(f"PRI per-emitter CSV saved to: {emit_path}")
        except Exception as e:
            print(f"Error saving PRI per-emitter CSV: {e}")

        # Frame summary CSV
        frame_fields = [
            "frame_id",
            "num_emitters_total",
            "num_emitters_analyzed",
            "pri_mae_us",
        ]
        try:
            write_header_f = not os.path.exists(frame_path)
            with open(frame_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=frame_fields)
                if write_header_f:
                    writer.writeheader()
                writer.writerow(frame_summary)
            print(f"PRI frame-summary CSV saved to: {frame_path}")
        except Exception as e:
            print(f"Error saving PRI frame-summary CSV: {e}")

    return {
        "emitters": per_emitter_rows,
        "frame_summary": frame_summary,
    }

def main():
    """Example usage of the inference class"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Radar deinterleaving inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data (.npy file)')
    parser.add_argument('--labels', type=str, 
                       help='Path to test labels (.npy file) - optional')
    parser.add_argument('--frame_idx', type=int, default=0,
                       help='Frame index to test (for single frame mode)')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Starting frame index for batch evaluation')
    parser.add_argument('--end-frame', type=int, default=None,
                       help='Ending frame index for batch evaluation')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Binary threshold')
    parser.add_argument('--save_plot', type=str,
                       help='Path to save plot (for single frame mode)')
    parser.add_argument('--batch-eval', action='store_true',
                       help='Run batch evaluation instead of single frame')
    parser.add_argument('--enable-snr', action='store_true',
                       help='Enable SNR calculation during batch evaluation')
    parser.add_argument('--ground-truth', type=str,
                       help='Path to ground truth data for SNR calculation')
    parser.add_argument('--save-csv', type=str,
                       help='Path to save CSV file with frame-by-frame results')
    parser.add_argument('--enable-pri', action='store_true',
                       help='Run PRI analysis per emitter after inference')
    parser.add_argument('--pri-model', type=str,
                       default=os.path.join('PRI model', 'pri_model', 'best_model.pt'),
                       help='Path to PRI model checkpoint (.pt)')
    parser.add_argument('--pri-gt-json', type=str,
                       help='Path to PRI ground truth JSON file (e.g., pri_information_GT.json)')
    parser.add_argument('--pri-fs-hz', type=float, default=3_000_000.0,
                       help='Sampling frequency for PRI analysis (Hz)')
    parser.add_argument('--pri-save-csv', type=str,
                       help='Prefix path to save PRI analysis CSV files '
                            '(will create <prefix>_emitters.csv and <prefix>_frames.csv)')
    parser.add_argument('--f1-threshold', type=float, default=0.0,
                       help='Minimum F1 score threshold for plotting emitters (default: 0.0, plots all)')
    
    args = parser.parse_args()
    
    # Load data
    X = np.load(args.data)
    if args.labels:
        Y = np.load(args.labels)
    else:
        Y = None
    
    # Initialize inference
    inference = RadarDeinterleavingInference(args.checkpoint)
    
    # Set normalization stats (using first few samples as proxy)
    inference.set_normalization_stats(X[:10])
    
    if args.batch_eval:
        # Batch evaluation mode
        if Y is None:
            print("Error: Labels are required for batch evaluation")
            return
        
        # Load ground truth data if SNR is enabled
        gt_data = None
        if args.enable_snr:
            if args.ground_truth is None:
                print("Error: Ground truth data path is required when --enable-snr is used")
                return
            try:
                gt_data = np.load(args.ground_truth)
                print(f"Ground truth data loaded: {gt_data.shape}")
            except Exception as e:
                print(f"Error loading ground truth data: {e}")
                return
        
        # If PRI flags are provided, inject them into config for batch mode
        if args.enable_pri:
            if inference.config is None:
                inference.config = {}
            pri_cfg = inference.config.get("pri", {})
            pri_cfg["enable_pri"] = True
            pri_cfg["pri_model_ckpt"] = args.pri_model
            pri_cfg["pri_gt_json"] = args.pri_gt_json
            pri_cfg["pri_fs_hz"] = args.pri_fs_hz
            inference.config["pri"] = pri_cfg

        print("Running batch evaluation...")
        metrics = inference.evaluate_batch(X, Y, args.start_frame, args.end_frame, args.threshold, 
                                         enable_snr=args.enable_snr, gt_data=gt_data, save_csv=args.save_csv)
        
        print("\n" + "="*50)
        print("BATCH EVALUATION RESULTS")
        print("="*50)
        print(f"Frames evaluated: {metrics['num_frames']}")
        print(f"Total assignments: {metrics['num_assignments']}")
        print(f"Average Dice Score: {metrics['dice']:.4f}")
        print(f"Average Precision: {metrics['precision']:.4f}")
        print(f"Average Recall: {metrics['recall']:.4f}")
        print(f"Average F1 Score: {metrics['f1']:.4f}")
        if 'snr' in metrics:
            print(f"Average SNR: {metrics['snr']:.2f} dB")
        if 'pri_mae_us' in metrics:
            print(f"Average PRI MAE: {metrics['pri_mae_us']:.2f} us")
        if 'pri_cls_acc' in metrics:
            print(f"Average PRI classification accuracy: {metrics['pri_cls_acc']:.3f}")
        print("="*50)
        
    else:
        # Single frame mode (original behavior)
        if Y is not None:
            gt_masks = Y[args.frame_idx]
        else:
            gt_masks = None
        
        # Get I/Q data for the frame
        I_data = X[args.frame_idx, 0, :]
        Q_data = X[args.frame_idx, 1, :]
        
        # Predict
        if gt_masks is not None:
            result = inference.predict_with_pit(I_data, Q_data, gt_masks, args.threshold)
            print("PIT assignment applied")
        else:
            result = inference.predict(I_data, Q_data, args.threshold)
            print("Basic prediction (no PIT)")

        # PRI analysis (per emitter)
        if args.enable_pri:
            if not os.path.exists(args.pri_model):
                print(f"Warning: PRI model checkpoint not found: {args.pri_model}")
            else:
                run_pri_analysis(
                    I_data,
                    Q_data,
                    result,
                    pri_model_ckpt=args.pri_model,
                    fs_hz=args.pri_fs_hz,
                    gt_json_path=args.pri_gt_json,
                    frame_idx=args.frame_idx,
                    csv_prefix=args.pri_save_csv,
                )
        
        # Print windowing info if used
        if 'windowing_info' in result:
            info = result['windowing_info']
            print(f"Windowing used: {info['num_windows']} windows of length {info['window_len']}")
        
        # Visualize
        inference.visualize_prediction(I_data, Q_data, result, gt_masks, 
                                     save_path=args.save_plot, show_plot=True,
                                     f1_threshold=args.f1_threshold)
        
        print("Inference completed!")

if __name__ == "__main__":
    main()
