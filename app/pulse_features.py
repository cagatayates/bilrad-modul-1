from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

DEFAULT_FS_HZ = 3_000_000.0


def _mask_to_segments(mask_1d: np.ndarray, min_len: int, merge_gap: int) -> list[tuple[int, int]]:
    """Convert boolean mask to inclusive [start, end] segments."""
    m = np.asarray(mask_1d).astype(float).ravel() > 0.5
    if m.size == 0:
        return []

    x = np.concatenate([[0], m.astype(np.int8), [0]])
    d = np.diff(x)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    segs = [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_len]

    if merge_gap <= 0 or len(segs) <= 1:
        return segs

    merged: list[tuple[int, int]] = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s - pe - 1 <= merge_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s + 1 >= min_len]


def _pri_stats_from_starts(starts: np.ndarray) -> tuple[float, float, float, float]:
    if starts.size < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    pri = np.diff(starts.astype(np.float64))
    return (
        float(np.mean(pri)),
        float(np.std(pri)),
        float(np.min(pri)),
        float(np.max(pri)),
    )


def _spectral_features_from_mag(mag: np.ndarray, fs_hz: float) -> dict[str, float]:
    """
    Frequency-domain features computed from magnitude samples.
    Units are in Hz using the provided sampling frequency (fs_hz).
    """
    x = np.asarray(mag, dtype=np.float64).ravel()
    if x.size < 8:
        return {
            "spec_peak_hz": float("nan"),
            "spec_centroid_hz": float("nan"),
            "spec_bandwidth_hz": float("nan"),
            "spec_flatness": float("nan"),
            "spec_rolloff_hz_95": float("nan"),
        }

    x = x - float(np.mean(x))
    w = np.hanning(x.size)
    xw = x * w
    X = np.fft.rfft(xw)
    P = (np.abs(X) ** 2).astype(np.float64)
    fs = float(fs_hz) if fs_hz is not None else DEFAULT_FS_HZ
    if not np.isfinite(fs) or fs <= 0:
        fs = DEFAULT_FS_HZ
    freqs = np.fft.rfftfreq(xw.size, d=1.0 / fs)  # Hz

    # Ignore DC for peak frequency if possible
    if P.size > 1:
        P_no_dc = P.copy()
        P_no_dc[0] = 0.0
    else:
        P_no_dc = P

    total = float(P.sum())
    if total <= 0.0:
        return {
            "spec_peak_hz": float("nan"),
            "spec_centroid_hz": float("nan"),
            "spec_bandwidth_hz": float("nan"),
            "spec_flatness": float("nan"),
            "spec_rolloff_hz_95": float("nan"),
        }

    peak_idx = int(np.argmax(P_no_dc))
    peak_f = float(freqs[peak_idx])

    centroid = float((freqs * P).sum() / total)
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * P).sum() / total))

    # Spectral flatness (geometric mean / arithmetic mean) of power spectrum
    eps = 1e-12
    flatness = float(np.exp(np.mean(np.log(P + eps))) / (np.mean(P + eps)))

    # 95% spectral rolloff frequency
    cdf = np.cumsum(P) / total
    roll_idx = int(np.searchsorted(cdf, 0.95))
    roll_idx = min(max(roll_idx, 0), freqs.size - 1)
    roll_f = float(freqs[roll_idx])

    return {
        "spec_peak_hz": peak_f,
        "spec_centroid_hz": centroid,
        "spec_bandwidth_hz": bandwidth,
        "spec_flatness": flatness,
        "spec_rolloff_hz_95": roll_f,
    }


def compute_emitter_pulse_words_table(
    I: np.ndarray,
    Q: np.ndarray,
    masks_display: np.ndarray,
    min_pulse_len: int = 5,
    merge_gap: int = 0,
    fs_hz: float = DEFAULT_FS_HZ,
) -> pd.DataFrame:
    """
    Emitter başına örnek şablondaki darbe kelimeleri.

    Kolonlar: pulse_count, pw_*, pri_*, duty_cycle, mag_*, start/end konum istatistikleri,
    ve frekans-bazlı öznitelikler (FFT üzerinden, birim: Hz).
    """
    I = np.asarray(I, dtype=np.float64).ravel()
    Q = np.asarray(Q, dtype=np.float64).ravel()
    L = min(I.size, Q.size)
    I, Q = I[:L], Q[:L]
    mag = np.sqrt(I**2 + Q**2)

    masks = np.asarray(masks_display)
    if masks.ndim != 2:
        raise ValueError(f"masks_display must be (N, L). Got {masks.shape}")
    N, Lm = masks.shape
    L = min(L, Lm)
    I, Q, mag = I[:L], Q[:L], mag[:L]
    masks = masks[:, :L]
    fs = float(fs_hz) if fs_hz is not None else DEFAULT_FS_HZ
    if not np.isfinite(fs) or fs <= 0:
        fs = DEFAULT_FS_HZ
    samp_to_us = 1e6 / fs

    rows: list[dict[str, Any]] = []
    for em in range(N):
        mask_e = masks[em] > 0.5
        segs = _mask_to_segments(mask_e, min_len=min_pulse_len, merge_gap=merge_gap)
        pulse_count = len(segs)
        if pulse_count == 0:
            rows.append(
                {
                    "emitter": em + 1,
                    "pulse_count": 0,
                    "pw_mean": np.nan,
                    "pw_std": np.nan,
                    "pw_min": np.nan,
                    "pw_max": np.nan,
                    "pw_mean_us": np.nan,
                    "pw_std_us": np.nan,
                    "pw_min_us": np.nan,
                    "pw_max_us": np.nan,
                    "pri_mean": np.nan,
                    "pri_std": np.nan,
                    "pri_min": np.nan,
                    "pri_max": np.nan,
                    "pri_mean_us": np.nan,
                    "pri_std_us": np.nan,
                    "pri_min_us": np.nan,
                    "pri_max_us": np.nan,
                    "duty_cycle": 0.0,
                    "mag_mean_pulse": np.nan,
                    "mag_max_pulse": np.nan,
                    "pulse_start_mean": np.nan,
                    "pulse_start_std": np.nan,
                    "pulse_end_mean": np.nan,
                    "pulse_end_std": np.nan,
                    "pulse_start_mean_us": np.nan,
                    "pulse_start_std_us": np.nan,
                    "pulse_end_mean_us": np.nan,
                    "pulse_end_std_us": np.nan,
                    "spec_peak_hz": np.nan,
                    "spec_centroid_hz": np.nan,
                    "spec_bandwidth_hz": np.nan,
                    "spec_flatness": np.nan,
                    "spec_rolloff_hz_95": np.nan,
                }
            )
            continue

        widths = np.array([e - s + 1 for s, e in segs], dtype=np.float64)
        starts = np.array([s for s, _ in segs], dtype=np.float64)
        ends = np.array([e for _, e in segs], dtype=np.float64)
        pri_mean, pri_std, pri_min, pri_max = _pri_stats_from_starts(starts)

        duty = float(widths.sum() / max(L, 1))

        idx_list = [np.arange(s, e + 1) for s, e in segs]
        idx = np.concatenate(idx_list) if idx_list else np.array([], dtype=np.int64)
        mag_pulse = mag[idx] if idx.size else np.array([], dtype=np.float64)
        sf = _spectral_features_from_mag(mag_pulse, fs_hz=fs_hz) if mag_pulse.size else _spectral_features_from_mag(np.array([]), fs_hz=fs_hz)

        row: dict[str, Any] = {
            "emitter": em + 1,
            "pulse_count": int(pulse_count),
            "pw_mean": float(np.mean(widths)),
            "pw_std": float(np.std(widths)),
            "pw_min": float(np.min(widths)),
            "pw_max": float(np.max(widths)),
            "pw_mean_us": float(np.mean(widths) * samp_to_us),
            "pw_std_us": float(np.std(widths) * samp_to_us),
            "pw_min_us": float(np.min(widths) * samp_to_us),
            "pw_max_us": float(np.max(widths) * samp_to_us),
            "pri_mean": pri_mean,
            "pri_std": pri_std,
            "pri_min": pri_min,
            "pri_max": pri_max,
            "pri_mean_us": float(pri_mean * samp_to_us) if np.isfinite(pri_mean) else np.nan,
            "pri_std_us": float(pri_std * samp_to_us) if np.isfinite(pri_std) else np.nan,
            "pri_min_us": float(pri_min * samp_to_us) if np.isfinite(pri_min) else np.nan,
            "pri_max_us": float(pri_max * samp_to_us) if np.isfinite(pri_max) else np.nan,
            "duty_cycle": duty,
            "mag_mean_pulse": float(np.mean(mag_pulse)) if mag_pulse.size else np.nan,
            "mag_max_pulse": float(np.max(mag_pulse)) if mag_pulse.size else np.nan,
            "pulse_start_mean": float(np.mean(starts)),
            "pulse_start_std": float(np.std(starts)),
            "pulse_end_mean": float(np.mean(ends)),
            "pulse_end_std": float(np.std(ends)),
            "pulse_start_mean_us": float(np.mean(starts) * samp_to_us),
            "pulse_start_std_us": float(np.std(starts) * samp_to_us),
            "pulse_end_mean_us": float(np.mean(ends) * samp_to_us),
            "pulse_end_std_us": float(np.std(ends) * samp_to_us),
            **sf,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def format_pulse_words_table(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Round floats for display."""
    if df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == float or out[c].dtype == np.float64:
            out[c] = out[c].round(decimals)
    return out
