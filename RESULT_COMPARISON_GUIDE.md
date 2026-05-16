# Result Comparison Tool Guide

This guide explains how to use the `result_compare.py` script to analyze and compare performance metrics between different datasets.

## Overview

The result comparison tool allows you to:
1. Load results from two CSV files (main and denoised)
2. Sort the main results by SNR in ascending order
3. Extract frame IDs from a specified index range
4. Find corresponding metrics in the denoised results
5. Calculate and display average Dice Score and SNR

## Usage

### Basic Command
```bash
python result_compare.py --main-results results_test_1_main.csv \
                        --denoised-results results_test_1_denoised.csv \
                        --start-index 0 \
                        --end-index 50
```

### Command Line Arguments

- `--main-results PATH`: Path to main results CSV file (required)
- `--denoised-results PATH`: Path to denoised results CSV file (required)
- `--start-index N`: Starting index for frame selection (0-based, required)
- `--end-index N`: Ending index for frame selection (exclusive, required)
- `--verbose`: Enable detailed frame-by-frame output (optional)

## How It Works

### Step 1: Load and Sort Main Results
- Loads the main results CSV file
- Sorts by SNR in ascending order (smallest to largest)
- This creates a ranking from lowest to highest SNR

### Step 2: Extract Frame IDs
- Selects frames from the specified index range
- For example, `--start-index 0 --end-index 50` selects the first 50 frames (lowest SNR)

### Step 3: Find Corresponding Metrics
- Searches the denoised results for the same frame IDs
- Extracts Dice Score and SNR values for those frames

### Step 4: Calculate Averages
- Computes average Dice Score and SNR
- Provides additional statistics (min, max, standard deviation)

## Example Scenarios

### Scenario 1: Analyze Lowest SNR Frames
```bash
# Analyze the 100 frames with lowest SNR
python result_compare.py --main-results results_test_1_main.csv \
                        --denoised-results results_test_1_denoised.csv \
                        --start-index 0 \
                        --end-index 100
```

### Scenario 2: Analyze Middle Range SNR Frames
```bash
# Analyze frames 200-300 (middle range)
python result_compare.py --main-results results_test_1_main.csv \
                        --denoised-results results_test_1_denoised.csv \
                        --start-index 200 \
                        --end-index 300
```

### Scenario 3: Analyze Highest SNR Frames
```bash
# Analyze the last 50 frames (highest SNR)
python result_compare.py --main-results results_test_1_main.csv \
                        --denoised-results results_test_1_denoised.csv \
                        --start-index 450 \
                        --end-index 500
```

## Output Format

### Standard Output
```
============================================================
RESULT COMPARISON ANALYSIS
============================================================
Number of frames analyzed: 50
Frame IDs: [12, 45, 78, 123, ...]

DICE SCORE STATISTICS:
  Average: 0.8234
  Std Dev: 0.0456
  Min:     0.7123
  Max:     0.9123

SNR STATISTICS:
  Average: 15.67 dB
  Std Dev: 2.34 dB
  Min:     12.45 dB
  Max:     18.90 dB

============================================================
```

### Verbose Output (with --verbose)
```
DETAILED FRAME DATA:
----------------------------------------
Frame 12: Dice=0.8234, SNR=15.23 dB
Frame 45: Dice=0.7891, SNR=14.87 dB
Frame 78: Dice=0.8567, SNR=16.45 dB
...
```

## CSV File Requirements

### Required Columns
Both CSV files must contain:
- `frame_id`: Frame identifier (integer)
- `dice_score`: Dice score value (float)
- `snr`: SNR value in dB (float)

### Example CSV Format
```csv
frame_id,dice_score,snr
0,0.8234,15.23
1,0.7891,14.87
2,0.8567,16.45
...
```

## Use Cases

### 1. Performance Analysis by SNR Range
- Compare how denoising affects different SNR ranges
- Identify which SNR levels benefit most from denoising
- Analyze performance degradation at low SNR

### 2. Quality Assessment
- Evaluate the effectiveness of denoising algorithms
- Compare average performance metrics
- Identify frames that need special attention

### 3. Statistical Analysis
- Calculate performance statistics for specific frame subsets
- Compare variance in performance across different SNR ranges
- Generate reports for different analysis scenarios

## Error Handling

The script handles common errors:
- Missing CSV files
- Invalid index ranges
- Missing required columns
- Empty or corrupted data

## Tips for Effective Analysis

1. **Start with small ranges** to test your analysis approach
2. **Use verbose mode** to see individual frame details
3. **Check data quality** by examining the loaded CSV files
4. **Compare different ranges** to understand performance patterns
5. **Save results** for further analysis or reporting

## Example Workflow

1. **Generate CSV files** using the batch evaluation script:
   ```bash
   # Generate main results
   python inference.py --checkpoint model.pth --data main_data.npy --labels main_labels.npy --batch-eval --enable-snr --ground-truth main_gt.npy --save-csv results_test_1_main.csv
   
   # Generate denoised results
   python inference.py --checkpoint model.pth --data denoised_data.npy --labels denoised_labels.npy --batch-eval --enable-snr --ground-truth denoised_gt.npy --save-csv results_test_1_denoised.csv
   ```

2. **Run comparison analysis**:
   ```bash
   python result_compare.py --main-results results_test_1_main.csv --denoised-results results_test_1_denoised.csv --start-index 0 --end-index 100 --verbose
   ```

3. **Analyze results** and adjust parameters as needed for different SNR ranges.
