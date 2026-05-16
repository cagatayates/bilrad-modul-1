## Local Inference UI (Streamlit)

This is a minimal, local UI to run **frame-level inference** with the trained U-Net checkpoint and visualize results.

### What it does

- Load checkpoint (`.pth`)
- Load full-frame scenario data (`X.npy` with shape `(B, 2, L)`)
- Optionally load labels (`Y.npy` with shape `(B, N, L)`) to show **PIT assignment + Dice/F1**
- Select a frame and visualize:
  - I/Q + magnitude signal
  - Probability and binary-mask heatmaps (N×L)
  - Per-emitter detail plot

### Install

From the project root:

```bash
pip install -r requirements.txt
```

Charts use **Plotly** (interactive zoom, pan, reset) in the browser; Matplotlib is not required for the UI plots.

Install PyTorch with the right CUDA version (optional) as described in `requirements.txt`.

### Run

From the project root:

```bash
streamlit run streamlit_app.py
```

### Default paths used by the UI

- `checkpoints/unet1d_pit_case1_N3_5_emitters_best.pth` (default)
- `checkpoints/unet1d_pit_case1_N3_best.pth` (older)
- `data/deinterleaving_2025_10_27_case_2_scenario_data.npy`
- `data/deinterleaving_2025_10_27_case_2_scenario_labels.npy`
- `data/deinterleaving_2025_10_27_case_2_clean_data.npy` (optional)

