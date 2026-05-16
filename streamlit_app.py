from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from app.infer_service import (
    LoadedData,
    compute_snr_db,
    load_full_frame_data,
    load_inference,
    run_single_frame,
    set_normalization_from_dataset,
)
from app.pulse_features import compute_emitter_pulse_words_table, format_pulse_words_table


APP_TITLE = "BILRAD - Module 1 UI"

# Plotly: zoom, pan, box zoom, reset axes (modebar). scrollZoom: mouse tekerleği ile yakınlaştırma.
PLOTLY_CONFIG: dict[str, Any] = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

# A restrained, presentation-friendly palette
COL_SIGNAL_MAG = "#1f77b4"
COL_SIGNAL_I = "#2ca02c"
COL_SIGNAL_Q = "#d62728"
COL_PROB_LINE = "#1f77b4"
COL_MASK_LINE = "#ff7f0e"
COL_GT_LINE = "#d62728"


# Algorithm registry. Add new algorithms here as they come online.
# Each entry: display name, availability flag, short tagline, and a default
# checkpoint path (relative parts under the project root) for that algorithm.
ALGORITHMS: dict[str, dict[str, Any]] = {
    "unet": {
        "name": "U-Net (2D)",
        "available": True,
        "tagline": "Per-sample, per-emitter segmentation. PIT eşleştirme + Dice / F1 metrikleri.",
        "default_checkpoint_parts": ("checkpoints", "unet1d_pit_case1_N3_5_emitters_best.pth"),
    },
    "yolo": {
        "name": "YOLO",
        "available": False,
        "tagline": "Darbe tespiti (bounding-box tabanlı). Yakında.",
        "default_checkpoint_parts": ("checkpoints", "yolo_best.pt"),
    },
}


# Denoising-method registry. Currently display-only; selection does NOT affect
# inference yet. When a method is implemented, set available=True and wire it
# into the pre-processing step before I/Q is sent to the model.
DENOISING_METHODS: dict[str, dict[str, Any]] = {
    "none": {
        "name": "None",
        "available": True,
        "tagline": "Denoising uygulanmaz (ham I/Q doğrudan modele girer).",
    },
    "wavelet": {
        "name": "Wavelet",
        "available": False,
        "tagline": "Wavelet tabanlı gürültü giderme. Yakında.",
    },
    "cnn": {
        "name": "CNN",
        "available": False,
        "tagline": "CNN denoiser. Yakında.",
    },
    "statistical": {
        "name": "Statistical",
        "available": False,
        "tagline": "İstatistiksel filtre tabanlı gürültü giderme. Yakında.",
    },
}


def _default_path(*parts: str) -> str:
    return str((Path(__file__).parent / Path(*parts)).resolve())


def _pick_file_dialog(title: str, filetypes: list[tuple[str, str]]) -> Optional[str]:
    """Open a native (tkinter) file picker. Local-UI only; returns the picked
    path or None if the dialog was cancelled or tkinter isn't available."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return path or None
    except Exception:
        return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _frame_summary(result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["num_emitters"] = int(result.get("num_emitters", -1))
    win = result.get("windowing_info")
    if isinstance(win, dict):
        out["windowing"] = f"used ({win.get('num_windows')} windows, len={win.get('window_len')}, stride={win.get('window_stride')})"
    else:
        out["windowing"] = "off"
    out["threshold"] = _safe_float(result.get("threshold", 0.5), 0.5)

    info = result.get("assignment_info")
    if isinstance(info, dict) and "assignments" in info:
        assigns = info.get("assignments") or []
        out["active_emitters_gt"] = int(info.get("active_emitters", len(assigns)))
        if assigns:
            dices = [a.get("dice_score") for a in assigns if a.get("dice_score") is not None]
            f1s = [a.get("f1_score") for a in assigns if a.get("f1_score") is not None]
            out["dice_mean"] = float(np.mean(dices)) if dices else None
            out["f1_mean"] = float(np.mean(f1s)) if f1s else None
        else:
            out["dice_mean"] = None
            out["f1_mean"] = None
    return out


def _get_pulse_mask_for_snr(gt: Optional[np.ndarray], masks_display: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Prefer GT-based pulse mask if available, otherwise use predicted binary mask union.
    Returns (L,) boolean mask or None.
    """
    if gt is not None:
        try:
            gt_arr = np.asarray(gt)
            return (gt_arr.sum(axis=0) > 0).astype(bool)
        except Exception:
            pass
    if masks_display is not None:
        try:
            m = np.asarray(masks_display)
            return (m.sum(axis=0) > 0).astype(bool)
        except Exception:
            pass
    return None


def _assignments_df(result: dict[str, Any]) -> Optional[pd.DataFrame]:
    info = result.get("assignment_info")
    if not isinstance(info, dict):
        return None
    assigns = info.get("assignments")
    if not isinstance(assigns, list):
        return None
    if not assigns:
        return pd.DataFrame(columns=["pred_channel", "gt_channel", "dice_score", "f1_score"])
    return pd.DataFrame(assigns)


def _plotly_signal(I: np.ndarray, Q: np.ndarray, title: str) -> go.Figure:
    mag = np.sqrt(I.astype(np.float64) ** 2 + Q.astype(np.float64) ** 2)
    x = np.arange(I.shape[0], dtype=np.int64)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mag,
            mode="lines",
            name="Magnitude |I+jQ|",
            line=dict(color=COL_SIGNAL_MAG, width=1.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=I,
            mode="lines",
            name="I",
            line=dict(color=COL_SIGNAL_I, width=0.8),
            opacity=0.65,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Q,
            mode="lines",
            name="Q",
            line=dict(color=COL_SIGNAL_Q, width=0.8),
            opacity=0.65,
        )
    )
    fig.update_layout(
        title=dict(text=title),
        template="plotly_white",
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Sample",
        yaxis_title="Amplitude",
        margin=dict(l=60, r=30, t=60, b=50),
    )
    fig.update_xaxes(
        rangeslider=dict(visible=False),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="solid",
        spikethickness=1,
    )
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="solid", spikethickness=1)
    return fig


def _plotly_heatmaps(
    probs: np.ndarray,
    title_prefix: str,
    vmax_prob: float = 1.0,
) -> go.Figure:
    # probs: (N, L)
    N, L = probs.shape
    xs = np.arange(L)
    ys = np.arange(N)
    # Presentation-friendly labels
    y_tickvals = ys
    y_ticktext = [f"Emitter {i+1}" for i in ys]

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=probs,
                x=xs,
                y=ys,
                colorscale="Viridis",
                zmin=0.0,
                zmax=float(vmax_prob),
                colorbar=dict(title="p", len=0.7, y=0.5, yanchor="middle"),
                hovertemplate="Emitter %{y}<br>Sample=%{x}<br>p=%{z:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=f"{title_prefix} — Probability (N×L)"),
        template="plotly_white",
        height=560,
        margin=dict(l=70, r=30, t=60, b=50),
        hovermode="closest",
        xaxis_title="Sample",
        yaxis_title="Emitter channel",
        # nicer visual for emitter channels (top-to-bottom)
        yaxis=dict(autorange="reversed", tickmode="array", tickvals=y_tickvals, ticktext=y_ticktext),
    )
    fig.update_xaxes(
        rangeslider=dict(visible=False),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="solid",
        spikethickness=1,
    )
    return fig


def _plotly_emitter_lines(
    probs: np.ndarray,
    binary_masks: np.ndarray,
    emitter_idx: int,
    threshold: float,
    gt: Optional[np.ndarray] = None,
) -> go.Figure:
    L = probs.shape[1]
    x = np.arange(L, dtype=np.int64)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=probs[emitter_idx],
            mode="lines",
            name="probability",
            line=dict(color=COL_PROB_LINE, width=1.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=binary_masks[emitter_idx],
            mode="lines",
            name="binary mask",
            line=dict(color=COL_MASK_LINE, width=1.1),
        )
    )
    if gt is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=gt[emitter_idx],
                mode="lines",
                name="ground truth",
                line=dict(color=COL_GT_LINE, width=1.0, dash="dot"),
                opacity=0.75,
            )
        )
    # Threshold guide line
    fig.add_hline(
        y=float(threshold),
        line_width=1,
        line_dash="dash",
        line_color="#6b7280",
        annotation_text="threshold",
        annotation_position="top right",
        annotation_font_size=10,
    )
    fig.update_layout(
        title=dict(text=f"Emitter {emitter_idx + 1} — Detail (threshold={threshold:.3f})"),
        template="plotly_white",
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Sample",
        yaxis_title="Activity / p",
        yaxis=dict(range=[-0.05, 1.05]),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    fig.update_xaxes(
        rangeslider=dict(visible=False),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="solid",
        spikethickness=1,
    )
    return fig


@st.cache_resource(show_spinner=False)
def _cached_infer(checkpoint_path: str, device: str) -> Any:
    return load_inference(checkpoint_path, device=device)


@st.cache_data(show_spinner=False)
def _cached_data(x_path: str, y_path: Optional[str], clean_path: Optional[str], mmap: bool) -> LoadedData:
    return load_full_frame_data(x_path=x_path, y_path=y_path, clean_path=clean_path, mmap=mmap)


@st.cache_data(show_spinner=False)
def _cached_norm_stats(x_path: str, n_frames: int, seed: int, checkpoint_path: str, device: str) -> dict[str, float]:
    # Tie stats to both dataset and checkpoint/device; easiest correctness for v1.
    infer = _cached_infer(checkpoint_path, device)
    data = _cached_data(x_path, None, None, mmap=True)
    return set_normalization_from_dataset(infer, data.X, n_frames=n_frames, seed=seed)


def _infer_cache_key(
    checkpoint_path: str,
    device: str,
    x_path: str,
    y_path: Optional[str],
    clean_path: Optional[str],
    mmap: bool,
    norm_frames: int,
    frame_idx: int,
    threshold: float,
    use_windowing: bool,
) -> tuple:
    """Inputs that require re-running the model for this frame."""
    return (
        checkpoint_path,
        device,
        x_path,
        y_path or "",
        clean_path or "",
        mmap,
        int(norm_frames),
        int(frame_idx),
        float(threshold),
        bool(use_windowing),
    )


def _snapshot_result_for_ui(result: dict[str, Any]) -> dict[str, Any]:
    """Shallow copy with numpy arrays detached for session_state (no torch)."""
    out: dict[str, Any] = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            out[k] = np.array(v, copy=True)
        else:
            out[k] = v
    return out


def _run_and_store_bundle(
    infer: Any,
    data: LoadedData,
    frame_idx: int,
    threshold: float,
    use_windowing: bool,
    cache_key: tuple,
) -> dict[str, Any]:
    """Run inference for `frame_idx` and stash a UI bundle into session_state['infer_bundle']."""
    result = run_single_frame(
        infer=infer,
        data=data,
        frame_idx=int(frame_idx),
        threshold=float(threshold),
        use_windowing=bool(use_windowing),
    )

    I = np.asarray(data.X[int(frame_idx), 0, :])
    Q = np.asarray(data.X[int(frame_idx), 1, :])
    probs = np.asarray(result["probabilities"])
    masks = np.asarray(result["binary_masks"])
    gt = np.asarray(data.Y[int(frame_idx)]) if data.Y is not None else None

    if gt is not None and "matched_predictions" in result:
        try:
            probs_display = np.asarray(result["matched_predictions"])
            masks_display = (probs_display >= float(result.get("threshold", threshold))).astype(float)
        except Exception:
            probs_display, masks_display = probs, masks
    else:
        probs_display, masks_display = probs, masks

    L_out = int(probs_display.shape[1])
    I = I[:L_out]
    Q = Q[:L_out]
    if gt is not None:
        gt = gt[:, :L_out]

    bundle = {
        "key": cache_key,
        "I": I,
        "Q": Q,
        "gt": gt,
        "clean": (np.asarray(data.CLEAN[int(frame_idx)]) if data.CLEAN is not None else None),
        "probs_display": probs_display,
        "masks_display": masks_display,
        "result": _snapshot_result_for_ui(result),
        "frame_idx": int(frame_idx),
        "threshold_ui": float(threshold),
    }
    st.session_state["infer_bundle"] = bundle
    return bundle


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # Session-state defaults (idempotent on rerun)
    if "run_mode" not in st.session_state:
        st.session_state.run_mode = "Manual"
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "fps" not in st.session_state:
        st.session_state.fps = 2
    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0
    if "start_frame" not in st.session_state:
        st.session_state.start_frame = 0
    if "first_tick" not in st.session_state:
        st.session_state.first_tick = False
    if "algorithm" not in st.session_state:
        st.session_state.algorithm = "unet"
    if "denoising" not in st.session_state:
        st.session_state.denoising = "none"

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }
        .stMetric { background: rgba(250,250,250,0.65); border: 1px solid rgba(0,0,0,0.06); padding: 10px; border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(APP_TITLE)
    st.caption("Load a checkpoint + full-frame .npy data, select a frame, run inference, and visualize results.")

    with st.sidebar:
        # ---------- Algorithm selector (top-level) ----------
        st.markdown("### Algorithm")
        _algo_keys = list(ALGORITHMS.keys())

        def _fmt_algo(k: str) -> str:
            info = ALGORITHMS[k]
            return f"{info['name']}{'' if info['available'] else '  ·  🚧 Coming soon'}"

        _default_algo_index = (
            _algo_keys.index(st.session_state.algorithm)
            if st.session_state.algorithm in _algo_keys
            else 0
        )
        selected_algo = st.selectbox(
            "Algorithm",
            options=_algo_keys,
            index=_default_algo_index,
            format_func=_fmt_algo,
            label_visibility="collapsed",
            help="Aktif algoritma. Yeni algoritmalar eklendikçe bu menüde görünür.",
        )
        if selected_algo != st.session_state.algorithm:
            st.session_state.algorithm = selected_algo
            # Force a clean state when switching algorithms — old bundle is invalid,
            # and the checkpoint default should follow the new algorithm.
            st.session_state.pop("infer_bundle", None)
            st.session_state.pop("ckpt_path", None)
            st.session_state.playing = False

        algo_info = ALGORITHMS[selected_algo]
        st.caption(algo_info["tagline"])

        if not algo_info["available"]:
            st.warning(
                f"⚠️ **{algo_info['name']}** henüz hazır değil. "
                "Çalışan bir algoritma seçin (örn. **U-Net (2D)**)."
            )
            st.stop()

        # ---------- Denoising selector (display-only for now) ----------
        st.markdown("### Denoising")
        _dn_keys = list(DENOISING_METHODS.keys())

        def _fmt_denoising(k: str) -> str:
            info = DENOISING_METHODS[k]
            return f"{info['name']}{'' if info['available'] else '  ·  🚧 Coming soon'}"

        _default_dn_index = (
            _dn_keys.index(st.session_state.denoising)
            if st.session_state.denoising in _dn_keys
            else 0
        )
        selected_denoising = st.selectbox(
            "Denoising",
            options=_dn_keys,
            index=_default_dn_index,
            format_func=_fmt_denoising,
            label_visibility="collapsed",
            help=(
                "I/Q sinyaline uygulanacak gürültü giderme yöntemi. "
                "Şu an sadece görsel — seçim inference'i etkilemez."
            ),
        )
        if selected_denoising != st.session_state.denoising:
            st.session_state.denoising = selected_denoising

        dn_info = DENOISING_METHODS[selected_denoising]
        st.caption(dn_info["tagline"])

        if not dn_info["available"]:
            st.info(
                f"ℹ️ **{dn_info['name']}** denoising henüz uygulanmıyor. "
                "Inference şu an gürültü gidermesiz çalışıyor."
            )

        st.divider()

        # Initialize file-path session state (so the 📂 Browse buttons can write to them)
        if "ckpt_path" not in st.session_state:
            st.session_state.ckpt_path = _default_path(*algo_info["default_checkpoint_parts"])
        if "x_path" not in st.session_state:
            st.session_state.x_path = _default_path("data", "deinterleaving_2025_10_27_case_2_scenario_data.npy")
        if "y_path" not in st.session_state:
            st.session_state.y_path = _default_path("data", "deinterleaving_2025_10_27_case_2_scenario_labels.npy")
        if "clean_path" not in st.session_state:
            st.session_state.clean_path = _default_path("data", "deinterleaving_2025_10_27_case_2_clean_data.npy")

        def _path_row(
            label: str,
            state_key: str,
            dialog_title: str,
            filetypes: list[tuple[str, str]],
            help_text: str,
            button_key: str,
        ) -> str:
            pc1, pc2 = st.columns([3, 1], gap="small", vertical_alignment="bottom")
            with pc1:
                value = st.text_input(label, key=state_key, help=help_text)
            with pc2:
                if st.button("📂", key=button_key, width="stretch", help="Yerel dosya seç…"):
                    picked = _pick_file_dialog(dialog_title, filetypes)
                    if picked:
                        st.session_state[state_key] = picked
                        st.rerun()
            return value

        with st.expander("Inputs", expanded=False):
            checkpoint_path = _path_row(
                label="Model (.pth)",
                state_key="ckpt_path",
                dialog_title="Select model file",
                filetypes=[("PyTorch model", "*.pth *.pt"), ("All files", "*.*")],
                help_text="Model dosyası. Algoritma değiştiğinde varsayılan dosya adı güncellenir; istediğinizi seçebilirsiniz.",
                button_key="browse_ckpt",
            )
            x_path = _path_row(
                label="Scenario data X (.npy)  (B,2,L)",
                state_key="x_path",
                dialog_title="Select scenario data X (.npy)",
                filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")],
                help_text="Girdi I/Q verisi. Beklenen şekil: (B, 2, L).",
                button_key="browse_x",
            )
            y_path = _path_row(
                label="Labels Y (.npy)  optional (B,N,L)",
                state_key="y_path",
                dialog_title="Select labels Y (.npy)",
                filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")],
                help_text="Opsiyonel GT etiketleri. PIT + Dice / F1 için gerekir. Beklenen şekil: (B, N, L).",
                button_key="browse_y",
            )
            clean_path = _path_row(
                label="Clean data (.npy) optional",
                state_key="clean_path",
                dialog_title="Select clean data (.npy)",
                filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")],
                help_text="Opsiyonel temiz sinyal verisi. SNR metriği için kullanılır.",
                button_key="browse_clean",
            )

        with st.expander("Inference", expanded=True):
            device = st.selectbox(
                "Device",
                options=["auto", "cpu", "cuda"],
                index=0,
                help="auto: varsa CUDA, yoksa CPU.",
            )
            threshold = st.slider(
                "Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="Binary mask için eşik. Inference çıktısını ve Darbe kelimelerini etkiler.",
            )
            use_windowing = st.toggle(
                "Use windowing",
                value=False,
                help="Model pencereleme ile eğitildiyse kullanılabilir. Full-frame kullanımda genelde kapalı tutulur.",
            )

        with st.expander("Display", expanded=True):
            mmap = st.toggle(
                "Memory-map .npy (recommended)",
                value=True,
                help="Büyük .npy dosyaları RAM'e tamamen almadan okur.",
            )
            norm_frames = st.slider(
                "Normalization frames (subset)",
                min_value=5,
                max_value=200,
                value=50,
                step=5,
                help="Normalize istatistikleri (I/Q mean/std) bu kadar frame üzerinden hesaplanır.",
            )
            f1_filter = st.slider(
                "F1 filter (GT only)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                help="Sadece GT varsa: PIT eşleşmelerinde düşük F1'leri filtreler (aktif emitter listesi).",
            )
            vmax_prob = st.slider(
                "Probability heatmap vmax",
                min_value=0.2,
                max_value=1.0,
                value=1.0,
                step=0.05,
                help="Heatmap renk ölçeği üst sınırı. Düşürmek kontrastı artırır.",
            )
            st.divider()
            x_view_mode = st.selectbox(
                "X-range",
                options=["Full", "Window", "Custom"],
                index=0,
                help="Tüm grafiklerde ortak X aralığı (zoom/pan dışında).",
            )
            if x_view_mode == "Window":
                x_start = st.number_input("X start", min_value=0, value=0, step=1000, help="Başlangıç sample index.")
                x_len = st.number_input("Window length", min_value=100, value=5000, step=500, help="Gösterilecek sample sayısı.")
                x_end = int(x_start) + int(x_len)
            elif x_view_mode == "Custom":
                x_start = st.number_input("X start", min_value=0, value=0, step=1000)
                x_end = st.number_input("X end", min_value=1, value=45000, step=1000)
            else:
                x_start, x_end = 0, None

        with st.expander("Pulse words / Darbe kelimeleri", expanded=False):
            min_pulse_len = st.slider(
                "Min pulse length (samples)",
                min_value=1,
                max_value=500,
                value=5,
                step=1,
                help="Bu uzunluğun altındaki segmentler darbe sayılmaz.",
            )
            merge_gap = st.slider(
                "Merge gap (samples)",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
                help="Aradaki boşluk bu değerden küçükse segmentler birleştirilir.",
            )

        st.divider()
        run_mode = st.radio(
            "Run mode",
            options=["Manual", "Auto-run"],
            index=0 if st.session_state.run_mode == "Manual" else 1,
            horizontal=True,
            help="Manual: tek frame için Run / Refresh. Auto-run: frame'leri otomatik ilerletir.",
        )
        # Reset playing state if mode flipped
        if run_mode != st.session_state.run_mode:
            st.session_state.run_mode = run_mode
            if run_mode == "Manual":
                st.session_state.playing = False
        st.caption("Transport kontrolleri sayfanın üst kısmında.")

    # ---------- Load data + model (errors stop the app) ----------
    try:
        data = _cached_data(
            x_path=x_path,
            y_path=y_path.strip() or None,
            clean_path=clean_path.strip() or None,
            mmap=mmap,
        )
        B = int(data.X.shape[0])
        L = int(data.X.shape[2])
        N_y = int(data.Y.shape[1]) if data.Y is not None else None
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        st.stop()

    try:
        infer = _cached_infer(checkpoint_path=checkpoint_path, device=device)
        cfg = getattr(infer, "config", None) or {}
        mcfg = cfg.get("model", {})
        stats = _cached_norm_stats(
            x_path=x_path,
            n_frames=norm_frames,
            seed=42,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        st.stop()

    # Clamp persisted indices to the loaded dataset's range
    st.session_state.frame_idx = int(min(max(0, int(st.session_state.frame_idx)), max(0, B - 1)))
    st.session_state.start_frame = int(min(max(0, int(st.session_state.start_frame)), max(0, B - 1)))

    # ---------- Diagnostics (collapsed by default) ----------
    with st.expander("Diagnostics — data shapes, model config, normalization", expanded=False):
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown("**Data**")
            st.write(f"X: `{tuple(data.X.shape)}`")
            if data.Y is not None:
                st.write(f"Y: `{tuple(data.Y.shape)}`")
            st.write(f"Frames B = **{B}** · Length L = **{L}**")
            if N_y is not None:
                st.write(f"Emitter channels N = **{N_y}**")
        with d2:
            st.markdown("**Model**")
            st.write(f"`num_emitters` = `{mcfg.get('num_emitters')}`")
            st.write(f"`in_channels` = `{mcfg.get('in_channels')}`")
            st.write(f"`depth` = `{mcfg.get('depth')}`")
        with d3:
            st.markdown("**Normalization (I/Q)**")
            if stats:
                s1, s2 = st.columns(2)
                with s1:
                    st.metric("I mean", f"{stats.get('ch_0_mean', 0.0):.4f}")
                    st.metric("I std", f"{stats.get('ch_0_std', 0.0):.4f}")
                with s2:
                    st.metric("Q mean", f"{stats.get('ch_1_mean', 0.0):.4f}")
                    st.metric("Q std", f"{stats.get('ch_1_std', 0.0):.4f}")
            else:
                st.write("—")

    # ---------- Top transport bar (mode-aware) ----------
    run_btn = False
    play_btn = False
    pause_btn = False
    stop_btn = False

    if st.session_state.run_mode == "Manual":
        tc1, tc2 = st.columns([6, 1], gap="small", vertical_alignment="bottom")
        with tc1:
            new_frame = st.slider(
                "Frame index",
                min_value=0,
                max_value=max(0, B - 1),
                value=int(st.session_state.frame_idx),
                step=1,
                help="Manual modda incelenecek frame. Run / Refresh ile inference çalıştır.",
            )
            if int(new_frame) != int(st.session_state.frame_idx):
                st.session_state.frame_idx = int(new_frame)
        with tc2:
            run_btn = st.button(
                "▶ Run / Refresh",
                type="primary",
                width="stretch",
                help="Seçili frame için inference çalıştır.",
            )
    else:
        tc1, tc2, tc3, tc4, tc5, tc6 = st.columns(
            [2, 1, 1, 1, 2, 2], gap="small", vertical_alignment="bottom"
        )
        with tc1:
            new_start = st.number_input(
                "Start frame",
                min_value=0,
                max_value=max(0, B - 1),
                value=int(st.session_state.start_frame),
                step=1,
                disabled=bool(st.session_state.playing),
                help="Auto-run bu frame'den başlar. Pause sonrası mevcut frame ile güncellenir; Stop ile 0'a sıfırlanır.",
            )
            if int(new_start) != int(st.session_state.start_frame):
                st.session_state.start_frame = int(new_start)
        with tc2:
            play_btn = st.button(
                "▶ Play",
                width="stretch",
                disabled=bool(st.session_state.playing),
                help="Auto-run'ı **Start frame**'den başlatır.",
            )
        with tc3:
            pause_btn = st.button(
                "⏸ Pause",
                width="stretch",
                disabled=not bool(st.session_state.playing),
                help="Duraklat ve **Start frame** alanını mevcut frame'e güncelle.",
            )
        with tc4:
            stop_btn = st.button(
                "⏹ Stop",
                width="stretch",
                disabled=(not bool(st.session_state.playing)) and int(st.session_state.start_frame) == 0,
                help="Durdur ve **Start frame** alanını 0'a sıfırla.",
            )
        with tc5:
            fps_val = st.slider(
                "Speed (fps)",
                min_value=1,
                max_value=10,
                value=int(st.session_state.fps),
                step=1,
                help="Saniyedeki frame sayısı. CPU'da 1–3, CUDA'da yüksek.",
            )
            st.session_state.fps = int(fps_val)
        with tc6:
            _badge_interval = (1.0 / float(int(st.session_state.fps))) if bool(st.session_state.playing) else None

            def _render_live_badge() -> None:
                is_play = bool(st.session_state.playing)
                cur = int(st.session_state.frame_idx)
                total = max(0, int(B) - 1)
                if is_play:
                    bg = "linear-gradient(135deg, #16a34a 0%, #0ea5e9 100%)"
                    fg = "#ffffff"
                    border = "transparent"
                    prefix_label = "▶ LIVE"
                    sublabel = f"{int(st.session_state.fps)} fps"
                else:
                    bg = "#f1f5f9"
                    fg = "#334155"
                    border = "#e2e8f0"
                    prefix_label = "FRAME"
                    sublabel = "paused"
                st.markdown(
                    f"""
                    <div style="
                        background: {bg};
                        color: {fg};
                        padding: 8px 14px;
                        border-radius: 12px;
                        text-align: center;
                        font-weight: 700;
                        line-height: 1.15;
                        border: 1px solid {border};
                        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
                    ">
                        <div style="font-size: 10px; letter-spacing: 1.5px; opacity: 0.92;">{prefix_label}</div>
                        <div style="font-size: 26px; margin-top: 2px;">{cur} / {total}</div>
                        <div style="font-size: 10px; opacity: 0.80; margin-top: 2px;">{sublabel}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if hasattr(st, "fragment"):
                @st.fragment(run_every=_badge_interval)
                def _live_badge_fragment() -> None:
                    _render_live_badge()
                _live_badge_fragment()
            else:
                _render_live_badge()

    # ---------- Transport-button side effects ----------
    if play_btn:
        if not hasattr(st, "fragment"):
            st.error(
                "Auto-run için Streamlit ≥ 1.37 gerekli (st.fragment). "
                "Lütfen güncelleyin: `pip install --upgrade streamlit`."
            )
        else:
            st.session_state.frame_idx = int(st.session_state.start_frame)
            st.session_state.playing = True
            # First fragment call after Play should just render the start frame,
            # not advance — otherwise playback skips the start_frame visually.
            st.session_state.first_tick = True
            st.rerun()
    if pause_btn:
        st.session_state.start_frame = int(st.session_state.frame_idx)
        st.session_state.playing = False
        st.rerun()
    if stop_btn:
        st.session_state.start_frame = 0
        st.session_state.playing = False
        st.rerun()

    frame_idx = int(st.session_state.frame_idx)

    infer_key = _infer_cache_key(
        checkpoint_path,
        device,
        x_path,
        y_path.strip() or None,
        clean_path.strip() or None,
        mmap,
        norm_frames,
        int(frame_idx),
        float(threshold),
        bool(use_windowing),
    )

    bundle = st.session_state.get("infer_bundle")
    cache_hit = isinstance(bundle, dict) and bundle.get("key") == infer_key
    is_auto = st.session_state.run_mode == "Auto-run"
    is_playing = bool(st.session_state.playing) and is_auto

    if run_btn:
        # Manual mode: explicit Run / Refresh
        with st.spinner("Running inference for the selected frame..."):
            _run_and_store_bundle(infer, data, int(frame_idx), float(threshold), bool(use_windowing), infer_key)
        bundle = st.session_state["infer_bundle"]
    elif is_auto and (bundle is None or bundle.get("frame_idx") != int(frame_idx)):
        # Auto-run: seed a bundle for the current frame so Overview has something to show
        # immediately (even before the first timer tick), and so live-scrub feels responsive.
        with st.spinner("Computing current frame..."):
            _run_and_store_bundle(infer, data, int(frame_idx), float(threshold), bool(use_windowing), infer_key)
        bundle = st.session_state["infer_bundle"]
    elif cache_hit or (is_auto and bundle is not None):
        # Manual + cache hit, OR Auto-run with any prior bundle (relaxed key match while playing).
        pass
    else:
        st.info(
            "Bu frame ve inference ayarları için henüz çalıştırılmadı — "
            "**Manual** modda *Run / Refresh*'e, **Auto-run** modda **▶ Play**'e basın."
        )
        st.stop()

    I = bundle["I"]
    Q = bundle["Q"]
    gt = bundle["gt"]
    clean = bundle.get("clean")
    probs_display = bundle["probs_display"]
    masks_display = bundle["masks_display"]
    result = bundle["result"]

    # Optional SNR computation (requires clean data)
    snr_db: Optional[float] = None
    if clean is not None:
        try:
            pulse_mask = _get_pulse_mask_for_snr(gt, masks_display)
            snr_db = compute_snr_db(infer, I=I, Q=Q, clean_frame=clean, pulse_mask=pulse_mask)
        except Exception:
            snr_db = None

    # Optional F1 filtering (only affects “active emitter list” for detail plot)
    active_emitters: Optional[list[int]] = None
    if gt is not None and isinstance(result.get("assignment_info"), dict):
        df = _assignments_df(result)
        if df is not None and not df.empty and "gt_channel" in df.columns:
            df2 = df.copy()
            if "f1_score" in df2.columns:
                df2 = df2[df2["f1_score"].fillna(0.0) >= float(f1_filter)]
            active_emitters = sorted({int(x) for x in df2["gt_channel"].tolist()})

    # Layout: tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Emitter detail", "PIT / Metrics", "Darbe kelimeleri"])

    # Shared X-range slicing for plots (does not change metrics)
    L_plot = int(I.shape[0])
    xs0 = int(max(0, x_start))
    if x_end is None:
        xs1 = L_plot
    else:
        xs1 = int(min(max(xs0 + 1, int(x_end)), L_plot))

    I_plot = I[xs0:xs1]
    Q_plot = Q[xs0:xs1]
    probs_plot = probs_display[:, xs0:xs1]
    masks_plot = masks_display[:, xs0:xs1]
    gt_plot = gt[:, xs0:xs1] if gt is not None else None

    with tab1:
        st.subheader(
            "Signal + overview heatmaps",
            help=(
                "Grafikler etkileşimli: sağ üstteki araç çubuğunda yakınlaştırma, kaydırma (pan), "
                "dikdörtgen seçerek zoom ve eksenleri sıfırlama var. "
                "Fare tekerleği ile de yakınlaştırabilirsiniz."
            ),
        )

        def _render_overview_static() -> None:
            """Render Overview from the latest session_state bundle (used by fragment + paused)."""
            b = st.session_state.get("infer_bundle")
            if not b:
                st.info("No data to render.")
                return
            I_b = b["I"]
            Q_b = b["Q"]
            probs_b = b["probs_display"]
            cur_frame = int(b.get("frame_idx", 0))
            L_b = int(I_b.shape[0])
            xs0_b = int(max(0, x_start))
            xs1_b = L_b if x_end is None else int(min(max(xs0_b + 1, int(x_end)), L_b))
            I_p = I_b[xs0_b:xs1_b]
            Q_p = Q_b[xs0_b:xs1_b]
            probs_p = probs_b[:, xs0_b:xs1_b]
            title_suffix = f" (x={xs0_b}:{xs1_b})" if (xs0_b != 0 or xs1_b != L_b) else ""

            if st.session_state.run_mode == "Auto-run" and bool(st.session_state.playing):
                st.caption(
                    f"▶ **Auto-run** · Frame **{cur_frame + 1} / {int(B)}** · {int(st.session_state.fps)} fps"
                )

            fig1 = _plotly_signal(I_p, Q_p, title=f"Frame {cur_frame} — Input signal{title_suffix}")
            st.plotly_chart(fig1, width="stretch", config=PLOTLY_CONFIG)
            fig2 = _plotly_heatmaps(
                probs=probs_p,
                title_prefix=f"Frame {cur_frame} — Model output{title_suffix}",
                vmax_prob=float(vmax_prob),
            )
            st.plotly_chart(fig2, width="stretch", config=PLOTLY_CONFIG)

        if is_playing and hasattr(st, "fragment"):
            play_interval = 1.0 / float(int(st.session_state.fps))

            @st.fragment(run_every=play_interval)
            def _overview_live() -> None:
                if st.session_state.run_mode == "Auto-run" and bool(st.session_state.playing):
                    if st.session_state.get("first_tick", False):
                        # Initial call right after Play: render the already-seeded start frame
                        # without advancing, so the user actually sees start_frame in the plot.
                        st.session_state.first_tick = False
                    else:
                        cur = int(st.session_state.frame_idx)
                        new_idx = (cur + 1) % int(B)
                        st.session_state.frame_idx = new_idx
                        new_key = _infer_cache_key(
                            checkpoint_path,
                            device,
                            x_path,
                            y_path.strip() or None,
                            clean_path.strip() or None,
                            mmap,
                            norm_frames,
                            new_idx,
                            float(threshold),
                            bool(use_windowing),
                        )
                        _run_and_store_bundle(
                            infer, data, new_idx, float(threshold), bool(use_windowing), new_key
                        )
                _render_overview_static()

            _overview_live()
        else:
            _render_overview_static()

    with tab2:
        st.subheader(
            "Emitter detail view",
            help="Aynı araç çubuğu ile zaman ekseninde zoom / pan yapılabilir. Threshold ve GT (varsa) ile birlikte görselleştirilir.",
        )
        if is_playing:
            st.info(
                "⏸ **Auto-run aktif** — emitter detayları yalnızca **Pause / Stop** sonrası tazelenir."
            )
        else:
            N = int(probs_display.shape[0])
            show_active_only = st.toggle(
                "Active emitters only (GT/PIT)",
                value=bool(active_emitters),
                help="GT varsa: PIT eşleşmesi olan emitter'ları gösterir. Yoksa tüm kanalları gösterir.",
            )
            if show_active_only and active_emitters:
                options = list(active_emitters)
                st.caption(f"Active emitters: {[i+1 for i in options]}")
            else:
                options = list(range(N))
            emitter_idx = st.selectbox(
                "Emitter channel",
                options=options,
                index=0,
                format_func=lambda i: f"Emitter {int(i)+1}",
            )
            fig3 = _plotly_emitter_lines(
                probs=probs_plot,
                binary_masks=masks_plot,
                emitter_idx=int(emitter_idx),
                threshold=float(result.get("threshold", threshold)),
                gt=gt_plot,
            )
            st.plotly_chart(fig3, width="stretch", config=PLOTLY_CONFIG)

    with tab3:
        st.subheader(
            "PIT assignment and metrics",
            help=(
                "GT (Y.npy) varsa: PIT eşleştirmesi ile per-emitter Dice / F1 hesaplanır; "
                "Clean data varsa SNR (dB) eklenir. Tablo CSV olarak indirilebilir."
            ),
        )
        if is_playing:
            st.info(
                "⏸ **Auto-run aktif** — PIT eşleşmeleri ve metrikler yalnızca **Pause / Stop** sonrası tazelenir."
            )
        else:
            summ = _frame_summary(result)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("num_emitters", str(summ.get("num_emitters")))
            c2.metric("windowing", str(summ.get("windowing")))
            c3.metric("dice_mean", f"{summ.get('dice_mean'):.4f}" if summ.get("dice_mean") is not None else "—")
            c4.metric("f1_mean", f"{summ.get('f1_mean'):.4f}" if summ.get("f1_mean") is not None else "—")
            c5.metric("snr (dB)", f"{snr_db:.2f}" if snr_db is not None else "—")

            df = _assignments_df(result)
            if df is None:
                st.info("No ground-truth / PIT assignment info available (labels not loaded).")
            else:
                if "f1_score" in df.columns:
                    df = df.sort_values(["f1_score", "dice_score"], ascending=False)
                st.dataframe(df, width="stretch", hide_index=True)
                st.download_button(
                    "Download PIT table (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"pit_metrics_frame_{int(frame_idx)}.csv",
                    mime="text/csv",
                    width="stretch",
                )

    with tab4:
        st.subheader(
            "Darbe tanımlama kelimeleri (emitter bazlı)",
            help=(
                "Tahmini binary mask üzerinden darbe segmentleri çıkarılır. "
                "PW ve PRI birimleri örnek (sample) indeksi; PRI ardışık darbe başlangıçları arasındaki farktır. "
                "Frekans öznitelikleri, darbe örneklerinden çıkarılan |I+jQ| üzerinden FFT ile hesaplanır (varsayılan fs = 3 MHz, birim: Hz)."
            ),
        )
        if is_playing:
            st.info(
                "⏸ **Auto-run aktif** — darbe kelimeleri tablosu yalnızca **Pause / Stop** sonrası tazelenir."
            )
            st.stop()
        try:
            df_pw = compute_emitter_pulse_words_table(
                I,
                Q,
                masks_display,
                min_pulse_len=int(min_pulse_len),
                merge_gap=int(merge_gap),
                fs_hz=3_000_000.0,
            )
            df_show = format_pulse_words_table(df_pw, decimals=4)
            # Keep time-unit (µs) columns; drop raw sample-unit columns + duty cycle for clarity.
            drop_cols = [
                "pw_mean",
                "pw_std",
                "pw_min",
                "pw_max",
                "pri_mean",
                "pri_std",
                "pri_min",
                "pri_max",
                "pulse_start_mean",
                "pulse_start_std",
                "pulse_end_mean",
                "pulse_end_std",
                "duty_cycle",
            ]
            df_show = df_show.drop(columns=[c for c in drop_cols if c in df_show.columns])
            df_show = df_show.rename(
                columns={
                    "emitter": "Emitter",
                    "pulse_count": "Darbe sayısı",
                    "pw_mean_us": "PW ort (µs)",
                    "pw_std_us": "PW std (µs)",
                    "pw_min_us": "PW min (µs)",
                    "pw_max_us": "PW max (µs)",
                    "pri_mean_us": "PRI ort (µs)",
                    "pri_std_us": "PRI std (µs)",
                    "pri_min_us": "PRI min (µs)",
                    "pri_max_us": "PRI max (µs)",
                    "mag_mean_pulse": "Ort |sinyal| (darbe)",
                    "mag_max_pulse": "Max |sinyal| (darbe)",
                    "pulse_start_mean_us": "Darbe başlangıç ort (µs)",
                    "pulse_start_std_us": "Darbe başlangıç std (µs)",
                    "pulse_end_mean_us": "Darbe bitiş ort (µs)",
                    "pulse_end_std_us": "Darbe bitiş std (µs)",
                    "spec_peak_hz": "Spektral tepe f (Hz)",
                    "spec_centroid_hz": "Spektral centroid (Hz)",
                    "spec_bandwidth_hz": "Spektral bant genişliği (Hz)",
                    "spec_flatness": "Spektral flatness",
                    "spec_rolloff_hz_95": "Rolloff f (95%) (Hz)",
                }
            )
            st.dataframe(df_show, width="stretch", hide_index=True)
            st.download_button(
                "Download pulse-words table (CSV)",
                data=df_show.to_csv(index=False).encode("utf-8"),
                file_name=f"pulse_words_frame_{int(frame_idx)}.csv",
                mime="text/csv",
                width="stretch",
            )
        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()

