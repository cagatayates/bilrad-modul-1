import numpy as np
from typing import Dict, Tuple, Optional

# =========================
# Yardımcılar
# =========================
def _moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    k = int(k)
    pad = (k - 1) // 2
    xpad = np.pad(x, (pad, k - 1 - pad), mode="reflect")
    ker = np.ones(k, dtype=np.float32) / k
    return np.convolve(xpad, ker, mode="valid")

def _remove_short_segments(mask: np.ndarray, min_len: int) -> np.ndarray:
    y = mask.copy()
    in_run = False
    run_start = 0
    for i, v in enumerate(mask.tolist() + [False]):  # sentinel
        if v and not in_run:
            in_run = True
            run_start = i
        elif not v and in_run:
            in_run = False
            run_end = i
            if (run_end - run_start) < min_len:
                y[run_start:run_end] = False
    return y

def _merge_close_segments(mask: np.ndarray, min_gap: int) -> np.ndarray:
    y = mask.copy()
    i = 0
    L = len(mask)
    while i < L:
        if not y[i]:
            i += 1
            continue
        j = i + 1
        while j < L and y[j]:
            j += 1
        k = j
        while k < L and not y[k]:
            k += 1
        if k < L and (k - j) <= min_gap:
            y[j:k] = True
            i = k
        else:
            i = j
    return y

def _hysteresis_mask(env: np.ndarray, hi: float, lo: float, min_len: int, min_gap: int) -> np.ndarray:
    above_hi = env >= hi
    above_lo = env >= lo

    mask = np.zeros_like(env, dtype=bool)
    active = False
    for i in range(len(env)):
        if not active:
            if above_hi[i]:
                active = True
                mask[i] = True
        else:
            mask[i] = True
            if not above_lo[i]:
                active = False

    if min_len > 1:
        mask = _remove_short_segments(mask, min_len)
    if min_gap > 0:
        mask = _merge_close_segments(mask, min_gap)
    return mask

def _auto_thresholds(env: np.ndarray, k_hi: float = 4.0, k_lo: float = 2.0) -> Tuple[float, float, float, float]:
    med = np.median(env)
    mad = np.median(np.abs(env - med)) + 1e-9
    hi = med + k_hi * mad
    lo = med + k_lo * mad
    if lo > hi:
        lo = 0.8 * hi
    return float(hi), float(lo), float(med), float(mad)

def _segments_from_mask(mask: np.ndarray):
    starts, ends = [], []
    in_run = False
    for i, v in enumerate(mask.tolist() + [False]):  # sentinel
        if v and not in_run:
            in_run = True
            starts.append(i)
        elif not v and in_run:
            in_run = False
            ends.append(i)
    return np.array(starts, dtype=int), np.array(ends, dtype=int)

def _ipi_from_toa(toa_samp: np.ndarray, fs_hz: float):
    toa_us = toa_samp.astype(np.float64) * 1e6 / float(fs_hz)
    ipi_us = np.diff(toa_us)
    ipi_us = ipi_us[np.isfinite(ipi_us)]
    ipi_us = ipi_us[ipi_us > 0]
    if ipi_us.size == 0:
        return toa_us, ipi_us
    p99 = np.percentile(ipi_us, 99.0)
    ipi_us = np.clip(ipi_us, 0, 2.0 * p99)
    return toa_us, ipi_us

def _histogram_1d(ipi_us: np.ndarray, nbins: int = 512, tmax_us: Optional[float] = 15000.0):
    """
    Varsayılan tmax_us = 15 ms (frame uzunluğuna uygun)
    """
    if ipi_us.size == 0:
        h = np.zeros(nbins, dtype=np.float32)
        edges = np.linspace(0, 1.0, nbins + 1, dtype=np.float32)
        return h, edges, 1.0
    if tmax_us is None:
        tmax_us = float(np.clip(1.5 * np.percentile(ipi_us, 95.0), 2000.0, 20000.0))
    edges = np.linspace(0.0, tmax_us, nbins + 1, dtype=np.float64)
    h, _ = np.histogram(ipi_us, bins=edges)
    h = h.astype(np.float32)
    s = h.sum()
    if s > 0:
        h /= s  # L1 normalize
    return h, edges.astype(np.float32), tmax_us

def _norm01(x: np.ndarray, eps: float = 1e-12):
    x = np.asarray(x, dtype=np.float64)
    mn, mx = np.min(x), np.max(x)
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)

def _acf_from_hist(hist: np.ndarray, nlags: int = 512) -> np.ndarray:
    """
    Histogram üzerinde otokorelasyon (merkezlenmiş ve normalize edilmiş).
    """
    h = hist.astype(np.float64)
    h = h - h.mean()
    den = (h @ h) + 1e-12
    corr = np.correlate(h, h, mode="full") / den
    mid = len(corr) // 2
    acf = corr[mid: mid + nlags]
    acf = np.clip(acf, -1.0, 1.0)
    return acf.astype(np.float32)

def _resample_1d(x: np.ndarray, out_len: int) -> np.ndarray:
    if x.size == 0:
        return np.zeros(out_len, dtype=np.float32)
    idx = np.linspace(0, len(x)-1, out_len, dtype=np.float64)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(x)-1)
    w = idx - i0
    y = (1.0 - w) * x[i0] + w * x[i1]
    return y.astype(np.float32)

def _linear_fit(y: np.ndarray) -> Tuple[float, float]:
    """
    y ~ a * t + b  (t = 0..len(y)-1)
    Dönen: (a, b)
    """
    n = len(y)
    if n < 2:
        return 0.0, float(y[0] if n == 1 else 0.0)
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    y_mean = y.mean()
    num = np.sum((t - t_mean) * (y - y_mean))
    den = np.sum((t - t_mean) ** 2) + 1e-12
    a = num / den
    b = y_mean - a * t_mean
    return float(a), float(b)

def _fft_top_freq(y: np.ndarray, fs_units: float) -> Tuple[float, float]:
    """
    Basit spektrum: y'de en baskın AC bileşenin frekansı ve güç oranı.
    fs_units: örnekleme 'pencere sırası' başına 1 birim ise 1.0 verin (PRI(t) serisi için).
    Döner: (f_peak, power_ratio)  -- DC hariç tepe / toplam güç
    """
    x = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    if x.size < 8 or np.allclose(x.std(), 0.0):
        return 0.0, 0.0
    n = int(2 ** np.ceil(np.log2(len(x))))
    spec = np.fft.rfft(x, n=n)
    power = (spec.real**2 + spec.imag**2)
    power[0] = 0.0  # DC hariç
    if power.sum() <= 0:
        return 0.0, 0.0
    k = int(np.argmax(power))
    f_peak = (k / n) * fs_units  # uygun birim (burada "per window" frekans)
    power_ratio = float(power[k] / (power.sum() + 1e-12))
    return float(f_peak), power_ratio

# =========================
# Ana API
# =========================
def extract_pri_features(
    iq: np.ndarray,
    fs_hz: float,
    smooth_len: int = 9,
    k_hi: float = 4.0,
    k_lo: float = 2.0,
    min_len_us: float = 5.0,
    min_gap_us: float = 2.0,
    nbins: int = 512,
    tmax_us: Optional[float] = 15000.0,
    # --- yeni opsiyonlar ---
    include_acf: bool = True,
    acf_nlags: int = 512,
    pri_window_pulses: int = 8,     # kayan pencerede kaç darbe ortalaması
    pri_hop_pulses: int = 4,        # pencere kaydırma
    pri_track_outlen: int = 64      # PRI(t) yeniden örnekleme uzunluğu
) -> Dict[str, np.ndarray | float | int]:
    """
    Tek verici I/Q ham verisinden:
      - TOA, IPI
      - Histogram (512, 0..15ms)
      - ACF (512) [opsiyonel]
      - PRI(t) kısa-istenen ortalama ve yeniden örneklenmiş izi (64)
      - İstatistik özetler + birleştirilmiş öznitelik vektörü (feat_vector) üretir.
    """
    x = np.asarray(iq)
    if x.ndim != 2:
        raise ValueError("iq must be 2D: (2,L) or (L,2)")
    if x.shape[0] != 2 and x.shape[1] == 2:
        x = x.T  # (2,L)
    elif x.shape[0] != 2 and x.shape[1] != 2:
        raise ValueError(f"Unexpected iq shape {x.shape}; expected (2,L) or (L,2)")

    I, Q = x[0].astype(np.float64), x[1].astype(np.float64)

    # 1) Enerji zarfı + maske
    env = np.sqrt(I * I + Q * Q)
    env = _moving_avg(env, k=smooth_len)
    hi, lo, med, mad = _auto_thresholds(env, k_hi=k_hi, k_lo=k_lo)
    min_len_samp = max(1, int(round(min_len_us * 1e-6 * fs_hz)))
    min_gap_samp = max(0, int(round(min_gap_us * 1e-6 * fs_hz)))
    mask = _hysteresis_mask(env, hi=hi, lo=lo, min_len=min_len_samp, min_gap=min_gap_samp)

    # 2) TOA & IPI
    starts, ends = _segments_from_mask(mask)
    toa_samp = starts
    toa_us, ipi_us = _ipi_from_toa(toa_samp, fs_hz=fs_hz)

    # 3) Histogram (0..tmax_us)
    hist, edges_us, tmax_used = _histogram_1d(ipi_us, nbins=nbins, tmax_us=tmax_us)

    # 4) ACF (histogram üzerinden)
    if include_acf:
        acf = _acf_from_hist(hist, nlags=acf_nlags)  # (512,)
        acf = _norm01(acf)  # 0..1 normalize (isteğe bağlı)
    else:
        acf = np.zeros(acf_nlags, dtype=np.float32)

    # 5) Kayan pencere ile PRI(t) izleme (pulses domain)
    # Her pencerede: o pencereye düşen IPI'ların medyanı (veya ortalaması)
    pri_track = []
    K = len(toa_us)
    w = max(2, int(pri_window_pulses))
    h = max(1, int(pri_hop_pulses))
    if K >= (w + 1):
        for s in range(0, K - w, h):
            # w pulse → (w-1) IPI
            ipi_win = np.diff(toa_us[s:s + w])
            if ipi_win.size > 0:
                pri_track.append(float(np.median(ipi_win)))
    pri_track = np.asarray(pri_track, dtype=np.float64)

    # 6) PRI(t) özetler + yeniden örnekleme
    if pri_track.size > 0:
        # trend (sliding -, 0, + için işaret/şiddet)
        slope, intercept = _linear_fit(pri_track)
        drift_ppm = (slope / (np.mean(pri_track) + 1e-9)) * 1e6  # ~ppm/step
        # wobulation (periodik modülasyon) — baskın AC frekansı ve güç oranı
        f_peak, pow_ratio = _fft_top_freq(pri_track, fs_units=1.0)  # per-hop
        # normalize edilebilir
        pri_track_norm = _norm01(pri_track)
        pri_track_resampled = _resample_1d(pri_track_norm, pri_track_outlen)
        pri_mean = float(np.mean(pri_track))
        pri_std = float(np.std(pri_track))
        pri_mad = float(np.median(np.abs(pri_track - np.median(pri_track))))
    else:
        slope = 0.0; drift_ppm = 0.0; f_peak = 0.0; pow_ratio = 0.0
        pri_track_resampled = np.zeros(pri_track_outlen, dtype=np.float32)
        pri_mean = 0.0; pri_std = 0.0; pri_mad = 0.0
    # 7) İstatistik özetler (hist + ipi + track)
    ipi_nonzero = ipi_us[ipi_us > 0] if ipi_us.size > 0 else np.array([], dtype=np.float64)
    if ipi_nonzero.size > 0:
        ipi_med = float(np.median(ipi_nonzero))
        ipi_mad = float(np.median(np.abs(ipi_nonzero - ipi_med)))
        ipi_std = float(np.std(ipi_nonzero))
        ipi_p10 = float(np.percentile(ipi_nonzero, 10.0))
        ipi_p90 = float(np.percentile(ipi_nonzero, 90.0))
        pulses = int(len(toa_us))
    else:
        ipi_med = ipi_mad = ipi_std = ipi_p10 = ipi_p90 = 0.0
        pulses = 0

    # 8) Öznitelik vektörü birleştirme
    # hist(512) || acf(512) || stats(18) || pri_track_resampled(64) = 1156
    stats_vec = np.array([
        ipi_med, ipi_mad, ipi_std, ipi_p10, ipi_p90,
        pri_mean, pri_std, pri_mad,
        slope, drift_ppm, f_peak, pow_ratio,
        float(pulses),
        float(tmax_used),
        float(med), float(mad),
        float(hi), float(lo),
    ], dtype=np.float32)

    feat_vector = np.concatenate([
        hist.astype(np.float32),
        acf.astype(np.float32),
        stats_vec,
        pri_track_resampled.astype(np.float32)
    ], axis=0).astype(np.float32)

    features = {
        "hist": hist.astype(np.float32),               # (512,)
        "acf": acf.astype(np.float32),                 # (512,)
        "feat_vector": feat_vector,                    # (1156,)
        "pri_track_us": pri_track.astype(np.float32),  # (~K', değişken)
        "pri_track_resampled": pri_track_resampled.astype(np.float32),  # (64,)
        "bin_edges_us": edges_us,                      # (513,)
        "tmax_us": float(tmax_used),
        "toa_samples": toa_samp.astype(np.int64),      # (K,)
        "toa_us": toa_us.astype(np.float64),           # (K,)
        "ipi_us": ipi_us.astype(np.float64),           # (K-1,)
        "env_median": float(med),
        "env_mad": float(mad),
        "hi_thresh": float(hi),
        "lo_thresh": float(lo),
        "smooth_len": int(smooth_len),
        "min_len_us": float(min_len_us),
        "min_gap_us": float(min_gap_us),
        "fs_hz": float(fs_hz),
        "stats": {
            "ipi_med": ipi_med, "ipi_mad": ipi_mad, "ipi_std": ipi_std,
            "ipi_p10": ipi_p10, "ipi_p90": ipi_p90,
            "pri_mean": pri_mean, "pri_std": pri_std, "pri_mad": pri_mad,
            "drift_slope_us_per_step": float(slope),
            "drift_ppm_per_step": float(drift_ppm),
            "wob_freq_per_step": float(f_peak),
            "wob_power_ratio": float(pow_ratio),
            "pulses": pulses
        }
    }
    return features
