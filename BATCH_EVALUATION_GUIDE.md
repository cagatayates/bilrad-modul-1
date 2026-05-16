# Batch Evaluation Guide

This guide explains how to use the new batch evaluation functionality in the radar deinterleaving inference script.

## Overview

The modified `inference.py` script now supports two modes:
1. **Single Frame Mode** (original): Test a single frame with visualization
2. **Batch Evaluation Mode** (new): Test multiple frames and calculate average performance metrics

## New Command Line Arguments

### Batch Evaluation Arguments
- `--batch-eval`: Enable batch evaluation mode (no visualization)
- `--start-frame N`: Starting frame index (default: 0)
- `--end-frame N`: Ending frame index (default: all frames)
- `--threshold F`: Binary threshold for mask conversion (default: 0.5)
- `--enable-snr`: Enable SNR calculation during batch evaluation
- `--ground-truth PATH`: Path to ground truth data for SNR calculation
- `--save-csv PATH`: Path to save CSV file with frame-by-frame results

### Existing Arguments (still work in single frame mode)
- `--checkpoint PATH`: Path to model checkpoint
- `--data PATH`: Path to test data (.npy file)
- `--labels PATH`: Path to test labels (.npy file)
- `--frame_idx N`: Frame index for single frame mode
- `--save_plot PATH`: Path to save plot (single frame mode only)

## Usage Examples

### 1. Batch Evaluation (New Feature)
```bash
# Evaluate frames 0-499 with performance metrics
python inference.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth \
                   --data data/deinterleaving_u_net_case_2_v0_scenario_data.npy \
                   --labels data/deinterleaving_u_net_case_2_v0_scenario_labels.npy \
                   --batch-eval \
                   --start-frame 0 \
                   --end-frame 500 \
                   --threshold 0.5
```

### 2. Batch Evaluation with SNR Calculation
```bash
# Evaluate frames 0-499 with performance metrics and SNR calculation
python inference.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth \
                   --data data/deinterleaving_u_net_test_1_verici_2_scenario_data.npy \
                   --labels data/deinterleaving_u_net_test_1_verici_2_scenario_labels.npy \
                   --batch-eval \
                   --start-frame 0 \
                   --end-frame 500 \
                   --threshold 0.5 \
                   --enable-snr \
                   --ground-truth data/deinterleaving_u_net_test_1_verici_2_ground_truth_data.npy
```

### 3. Batch Evaluation with SNR and CSV Export
```bash
# Evaluate frames 0-499 with performance metrics, SNR calculation, and CSV export
python inference.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth \
                   --data data/deinterleaving_u_net_test_1_verici_2_scenario_data.npy \
                   --labels data/deinterleaving_u_net_test_1_verici_2_scenario_labels.npy \
                   --batch-eval \
                   --start-frame 0 \
                   --end-frame 500 \
                   --threshold 0.5 \
                   --enable-snr \
                   --ground-truth data/deinterleaving_u_net_test_1_verici_2_ground_truth_data.npy \
                   --save-csv results_frame_metrics.csv
```

### 4. Single Frame Mode (Original)
```bash
# Test single frame with visualization
python inference.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth \
                   --data data/deinterleaving_u_net_case_2_v0_scenario_data.npy \
                   --labels data/deinterleaving_u_net_case_2_v0_scenario_labels.npy \
                   --frame_idx 0 \
                   --threshold 0.5
```

## Performance Metrics

The batch evaluation calculates and reports the following metrics:

1. **Dice Score**: Measures overlap between predicted and ground truth masks
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1 Score**: Harmonic mean of precision and recall
5. **SNR (Signal-to-Noise Ratio)**: Signal power / Noise power in dB (when enabled)

All metrics are calculated using PIT (Permutation Invariant Training) assignment to handle the permutation problem in multi-emitter scenarios.

### SNR Calculation
When `--enable-snr` is used, the system calculates SNR for each frame by:
- Using the ground truth data as the clean signal
- Calculating noise as the difference between input data and ground truth
- Computing SNR in dB: `10 * log10(signal_power / noise_power)`

## Output Format

### Without SNR
```
==================================================
BATCH EVALUATION RESULTS
==================================================
Frames evaluated: 500
Total assignments: 1250
Average Dice Score: 0.8234
Average Precision: 0.8567
Average Recall: 0.7891
Average F1 Score: 0.8212
==================================================
```

### With SNR
```
Evaluating frames 0 to 499...
Processing frame 0, dice score: 0.8234, SNR: 15.23 dB
Processing frame 1, dice score: 0.7891, SNR: 14.87 dB
Processing frame 2, dice score: 0.8567, SNR: 16.45 dB
...

==================================================
BATCH EVALUATION RESULTS
==================================================
Frames evaluated: 500
Total assignments: 1250
Average Dice Score: 0.8234
Average Precision: 0.8567
Average Recall: 0.7891
Average F1 Score: 0.8212
Average SNR: 15.67 dB
==================================================
```

## CSV Export Format

When using `--save-csv`, the system exports frame-by-frame results to a CSV file with the following columns:

### Without SNR
```csv
frame_id,dice_score
0,0.8234
1,0.7891
2,0.8567
...
```

### With SNR
```csv
frame_id,dice_score,snr
0,0.8234,15.23
1,0.7891,14.87
2,0.8567,16.45
...
```

The CSV file contains:
- **frame_id**: Frame index (0-based)
- **dice_score**: Average dice score for that frame
- **snr**: SNR value in dB (only when `--enable-snr` is used)

## Implementation Details

### New Methods Added

1. **`evaluate_batch()`**: Main batch evaluation method
   - Processes multiple frames sequentially
   - Applies PIT assignment for each frame
   - Calculates performance metrics
   - Returns average metrics across all frames

2. **Enhanced `main()`**: Modified to support both modes
   - Detects batch evaluation mode via `--batch-eval` flag
   - Handles new command line arguments
   - Provides clear output formatting

### Key Features

- **Progress Tracking**: Shows progress every 50 frames
- **PIT Assignment**: Uses Hungarian algorithm for optimal emitter assignment
- **Comprehensive Metrics**: Calculates Dice, Precision, Recall, and F1 scores
- **Memory Efficient**: Processes frames one at a time
- **Error Handling**: Graceful handling of edge cases

## Performance Considerations

- **Memory Usage**: Low memory footprint as frames are processed individually
- **Speed**: Progress tracking every 50 frames to monitor long-running evaluations
- **Accuracy**: Uses the same PIT assignment logic as training for consistent evaluation

## Troubleshooting

### Common Issues

1. **"Labels are required for batch evaluation"**
   - Solution: Provide `--labels` argument when using `--batch-eval`

2. **"Checkpoint file not found"**
   - Solution: Verify the checkpoint path is correct

3. **"Data file not found"**
   - Solution: Verify the data and labels file paths are correct

### Performance Tips

- Start with a small range (e.g., `--end-frame 10`) to test
- Use appropriate threshold values (0.3-0.7 typically work well)
- Monitor progress output for long-running evaluations

## Example Scripts

Two example scripts are provided:

1. **`test_batch_eval.py`**: Tests the functionality on first 5 frames
2. **`example_batch_eval.py`**: Shows how to run batch evaluation programmatically

Run these to verify the installation and functionality:

```bash
python test_batch_eval.py
python example_batch_eval.py
```
