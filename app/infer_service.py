from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from inference import RadarDeinterleavingInference


@dataclass(frozen=True)
class LoadedData:
    x_path: str
    y_path: Optional[str]
    clean_path: Optional[str]
    X: np.ndarray  # (B, 2, L)
    Y: Optional[np.ndarray]  # (B, N, L)
    CLEAN: Optional[np.ndarray]  # project-specific


def _as_path_str(p: str | Path) -> str:
    return str(Path(p).as_posix())


def load_inference(checkpoint_path: str, device: str = "auto") -> RadarDeinterleavingInference:
    if not checkpoint_path:
        raise ValueError("Checkpoint path is empty.")
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return RadarDeinterleavingInference(checkpoint_path=checkpoint_path, device=device)


def load_full_frame_data(
    x_path: str,
    y_path: Optional[str] = None,
    clean_path: Optional[str] = None,
    mmap: bool = True,
) -> LoadedData:
    if not x_path:
        raise ValueError("X data path is empty.")
    xp = Path(x_path)
    if not xp.exists():
        raise FileNotFoundError(f"X data not found: {x_path}")

    mmap_mode = "r" if mmap else None
    X = np.load(x_path, mmap_mode=mmap_mode)

    Y = None
    if y_path:
        yp = Path(y_path)
        if not yp.exists():
            raise FileNotFoundError(f"Y labels not found: {y_path}")
        Y = np.load(y_path, mmap_mode=mmap_mode)

    CLEAN = None
    if clean_path:
        cp = Path(clean_path)
        if not cp.exists():
            raise FileNotFoundError(f"Clean data not found: {clean_path}")
        CLEAN = np.load(clean_path, mmap_mode=mmap_mode)

    _validate_full_frame_shapes(X, Y)
    return LoadedData(
        x_path=_as_path_str(x_path),
        y_path=_as_path_str(y_path) if y_path else None,
        clean_path=_as_path_str(clean_path) if clean_path else None,
        X=X,
        Y=Y,
        CLEAN=CLEAN,
    )


def _validate_full_frame_shapes(X: np.ndarray, Y: Optional[np.ndarray]) -> None:
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (B, 2, L). Got ndim={X.ndim}, shape={getattr(X, 'shape', None)}")
    if X.shape[1] != 2:
        raise ValueError(f"Expected X with 2 channels (I/Q). Got X.shape[1]={X.shape[1]}. Full shape={X.shape}")

    if Y is None:
        return
    if not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array.")
    if Y.ndim != 3:
        raise ValueError(f"Expected Y with shape (B, N, L). Got ndim={Y.ndim}, shape={getattr(Y, 'shape', None)}")
    if Y.shape[0] != X.shape[0]:
        raise ValueError(f"X and Y must have same B. Got X.shape[0]={X.shape[0]}, Y.shape[0]={Y.shape[0]}")
    if Y.shape[2] != X.shape[2]:
        raise ValueError(f"X and Y must have same L. Got X.shape[2]={X.shape[2]}, Y.shape[2]={Y.shape[2]}")


def set_normalization_from_dataset(
    infer: RadarDeinterleavingInference,
    X: np.ndarray,
    n_frames: int = 50,
    seed: int = 42,
) -> dict[str, float]:
    """
    Compute per-channel mean/std from a subset of frames and set on inference object.
    Returns the computed stats for display.
    """
    B = int(X.shape[0])
    n = max(1, min(int(n_frames), B))
    rng = np.random.default_rng(seed)
    idx = rng.choice(B, size=n, replace=False) if B > n else np.arange(B)
    subset = np.asarray(X[idx])  # ensure materialized for stable stats on memmap

    infer.set_normalization_stats(subset)
    stats = infer.normalization_stats or {}
    return {k: float(v) for k, v in stats.items()}


def run_single_frame(
    infer: RadarDeinterleavingInference,
    data: LoadedData,
    frame_idx: int,
    threshold: float = 0.5,
    use_windowing: bool = False,
) -> dict[str, Any]:
    X = data.X
    if frame_idx < 0 or frame_idx >= X.shape[0]:
        raise IndexError(f"frame_idx out of range: {frame_idx} not in [0, {X.shape[0]-1}]")

    I = np.asarray(X[frame_idx, 0, :])
    Q = np.asarray(X[frame_idx, 1, :])

    if data.Y is not None:
        gt_masks = np.asarray(data.Y[frame_idx])
        return infer.predict_with_pit(I, Q, gt_masks=gt_masks, threshold=threshold, use_windowing=use_windowing)

    return infer.predict(I, Q, threshold=threshold, use_windowing=use_windowing)


def compute_snr_db(
    infer: RadarDeinterleavingInference,
    I: np.ndarray,
    Q: np.ndarray,
    clean_frame: np.ndarray,
    pulse_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute SNR (dB) using clean signal as reference.

    - clean_frame is expected as (2, L) or compatible.
    - noise is estimated as (input - clean).
    - If pulse_mask is provided, SNR is computed using pulse vs non-pulse regions.
    """
    clean = np.asarray(clean_frame)
    if clean.ndim != 2 or clean.shape[0] != 2:
        raise ValueError(f"Expected clean_frame shape (2, L). Got {getattr(clean, 'shape', None)}")

    # Align length
    L = min(int(I.shape[0]), int(Q.shape[0]), int(clean.shape[1]))
    I = np.asarray(I[:L], dtype=np.float64)
    Q = np.asarray(Q[:L], dtype=np.float64)
    clean = np.asarray(clean[:, :L], dtype=np.float64)

    input_signal = np.stack([I, Q], axis=0)  # (2, L)
    noise = input_signal - clean

    pm = None
    if pulse_mask is not None:
        pm = np.asarray(pulse_mask).astype(bool)
        if pm.ndim != 1:
            raise ValueError(f"Expected pulse_mask shape (L,). Got {getattr(pm, 'shape', None)}")
        pm = pm[:L]

    return float(infer.calculate_snr(clean, noise, pulse_mask=pm))

