# make_pri_dataset.py
import os
import json
import argparse
from pathlib import Path
import numpy as np

from pri_features import extract_pri_features

# ---- tqdm (progress bar) ----
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

FEATURE_CHOICES = ["hist", "acf", "pri_track", "pri_track_resampled", "feat_vector"]

def find_class_folders(root: Path):
    subs = [p for p in root.iterdir() if p.is_dir()]
    subs = sorted(subs, key=lambda p: p.name.lower())
    subs = [p for p in subs if any(child.suffix == ".npy" for child in p.glob("*.npy"))]
    return subs

def build_class_mapping(root: Path, classes_arg: list[str] | None):
    if classes_arg:
        classes = [c.strip() for c in classes_arg]
    else:
        subs = find_class_folders(root)
        if not subs:
            raise FileNotFoundError(
                f"No class subfolders with .npy files found under {root}. "
                f"Expected: root/staggered/*.npy, root/jittered/*.npy, ..."
            )
        classes = [p.name for p in subs]
    classes_sorted = sorted(classes, key=lambda s: s.lower())
    name_to_id = {name: idx for idx, name in enumerate(classes_sorted)}
    return classes_sorted, name_to_id

def collect_files_by_class(root: Path, classes_sorted: list[str]):
    buckets = []
    for cname in classes_sorted:
        cdir = root / cname
        if not cdir.is_dir():
            raise FileNotFoundError(f"Class folder not found: {cdir}")
        files = sorted([p for p in cdir.glob("*.npy")], key=lambda p: p.name)
        if not files:
            print(f"[Warn] Empty class folder (no .npy): {cdir}")
        buckets.append((cname, files))
    return buckets

def process_folder(
    in_dir: str,
    out_dir: str,
    fs_hz: float = 3_000_000,
    nbins: int = 512,
    tmax_us: float | None = 15000.0,
    smooth_len: int = 9,
    k_hi: float = 4.0,
    k_lo: float = 2.0,
    min_len_us: float = 5.0,
    min_gap_us: float = 2.0,
    include_acf: bool = True,
    acf_nlags: int = 512,
    pri_window_pulses: int = 8,
    pri_hop_pulses: int = 4,
    pri_track_outlen: int = 64,
    which_feature: str = "feat_vector",
    save_details: bool = True,
    classes: list[str] | None = None,
    shuffle: bool = False,
    seed: int = 42,
    # log-PRI etiketleri
    make_logpri: bool = True,
    min_pulses: int = 5,
    pri_label_json: str | None = None,
    logpri_eps_us: float = 1e-6,
    # progress bar kontrolü
    no_tqdm: bool = False,
):
    """
    Labeled dataset üretimi:
    - Klasör yapısı: in_dir/<class_name>/*.npy
    - X: seçilen öznitelik matrisi
    - y: sınıf id'leri
    - (ops.) y_logpri: log(PRI_us)  -- harici JSON yoksa IPI medyanından
    """
    if which_feature not in FEATURE_CHOICES:
        raise ValueError(f"`which_feature` must be one of {FEATURE_CHOICES}")

    in_root = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_dir = out_dir / "details"
    if save_details:
        details_dir.mkdir(parents=True, exist_ok=True)

    classes_sorted, name_to_id = build_class_mapping(in_root, classes)
    print(f"[Classes] {classes_sorted}")

    buckets = collect_files_by_class(in_root, classes_sorted)

    # Harici PRI etiketi haritası (opsiyonel, esnek eşleşme)
    pri_map = {}
    if pri_label_json:
        with open(pri_label_json, "r", encoding="utf-8") as fp:
            tmp = json.load(fp)
        for k, v in tmp.items():
            p = Path(k)
            pri_map[p.name] = float(v)                                   # sadece dosya adı
            pri_map[str(p)] = float(v)                                   # tam path
            pri_map[str(p.parent.name) + "/" + p.name] = float(v)        # class/file

    X_list, y_list, y_logpri_list = [], [], []
    meta = []

    kept = 0
    skipped = 0

    # Toplam dosya sayısı (progress bar için)
    total_files = sum(len(files) for _, files in buckets)
    use_bar = (tqdm is not None) and (not no_tqdm)

    pbar = tqdm(total=total_files, desc="Processing files", unit="file") if use_bar else None

    for cname, files in buckets:
        cid = name_to_id[cname]
        # İç döngü
        for f in files:
            rel = str(Path(cname) / f.name)
            arr = np.load(f)
            if arr.ndim != 2:
                raise RuntimeError(f"{f}: expected 2D (2,L) or (L,2), got {arr.shape}")
            if arr.shape[0] != 2 and arr.shape[1] == 2:
                arr = arr.T
            elif arr.shape[0] != 2 and arr.shape[1] != 2:
                raise RuntimeError(f"{f}: invalid shape {arr.shape}")

            feats = extract_pri_features(
                arr, fs_hz=fs_hz,
                smooth_len=smooth_len, k_hi=k_hi, k_lo=k_lo,
                min_len_us=min_len_us, min_gap_us=min_gap_us,
                nbins=nbins, tmax_us=tmax_us,
                include_acf=include_acf or (which_feature in ["acf", "feat_vector"]),
                acf_nlags=acf_nlags,
                pri_window_pulses=pri_window_pulses, pri_hop_pulses=pri_hop_pulses,
                pri_track_outlen=pri_track_outlen
            )

            # Minimum darbe sayısı kontrolü
            pulses = int(feats["stats"]["pulses"])
            if make_logpri and pulses < int(min_pulses):
                skipped += 1
                if save_details:
                    np.savez_compressed(
                        details_dir / f"{f.stem}_details.npz",
                        toa_samples=feats["toa_samples"],
                        toa_us=feats["toa_us"],
                        ipi_us=feats["ipi_us"],
                        bin_edges_us=feats["bin_edges_us"],
                        hist=feats["hist"],
                        acf=feats.get("acf", None),
                        pri_track_us=feats["pri_track_us"],
                        pri_track_resampled=feats["pri_track_resampled"]
                    )
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(cls=cname, pulses=pulses, kept=kept, skipped=skipped)
                continue

            # Özellik seçimi
            if which_feature == "hist":
                x = feats["hist"]
            elif which_feature == "acf":
                x = feats["acf"]
            elif which_feature in ["pri_track", "pri_track_resampled"]:
                x = feats["pri_track_resampled"]
            elif which_feature == "feat_vector":
                x = feats["feat_vector"]
            else:
                raise RuntimeError("unreachable")

            X_list.append(x)
            y_list.append(cid)

            # log-PRI ground truth
            logpri = None
            if make_logpri:
                pri_us = None
                if f.name in pri_map:
                    pri_us = pri_map[f.name]
                elif rel in pri_map:
                    pri_us = pri_map[rel]
                elif str(f) in pri_map:
                    pri_us = pri_map[str(f)]
                if pri_us is None:
                    pri_us = float(feats["stats"]["ipi_med"])
                    src = "ipi_median"
                else:
                    src = "json"
                pri_us = max(pri_us, logpri_eps_us)
                logpri = float(np.log(pri_us))
                y_logpri_list.append(logpri)
            else:
                src = None

            st = feats["stats"]
            meta.append({
                "file": rel,
                "class_name": cname,
                "class_id": int(cid),
                "num_samples": int(arr.shape[1]),
                "pulses": pulses,
                "ipi_med_us": float(st["ipi_med"]),
                "ipi_mad_us": float(st["ipi_mad"]),
                "ipi_std_us": float(st["ipi_std"]),
                "ipi_p10_us": float(st["ipi_p10"]),
                "ipi_p90_us": float(st["ipi_p90"]),
                "pri_mean_us": float(st["pri_mean"]),
                "pri_std_us": float(st["pri_std"]),
                "pri_mad_us": float(st["pri_mad"]),
                "drift_slope_us_per_step": float(st["drift_slope_us_per_step"]),
                "drift_ppm_per_step": float(st["drift_ppm_per_step"]),
                "wob_freq_per_step": float(st["wob_freq_per_step"]),
                "wob_power_ratio": float(st["wob_power_ratio"]),
                "tmax_us": float(feats["tmax_us"]),
                "env_median": float(feats["env_median"]),
                "env_mad": float(feats["env_mad"]),
                "hi_thresh": float(feats["hi_thresh"]),
                "lo_thresh": float(feats["lo_thresh"]),
                "logpri_label": logpri,
                "label_source": src,
            })

            if save_details:
                np.savez_compressed(
                    details_dir / f"{f.stem}_details.npz",
                    toa_samples=feats["toa_samples"],
                    toa_us=feats["toa_us"],
                    ipi_us=feats["ipi_us"],
                    bin_edges_us=feats["bin_edges_us"],
                    hist=feats["hist"],
                    acf=feats.get("acf", None),
                    pri_track_us=feats["pri_track_us"],
                    pri_track_resampled=feats["pri_track_resampled"]
                )
            kept += 1

            if pbar:
                pbar.update(1)
                pbar.set_postfix(cls=cname, pulses=pulses, kept=kept, skipped=skipped)

    if pbar:
        pbar.close()

    # Matrisler
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    if make_logpri:
        y_logpri = np.asarray(y_logpri_list, dtype=np.float32)

    # (Ops.) shuffle
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
        meta = [meta[i] for i in idx]
        if make_logpri:
            y_logpri = y_logpri[idx]

    # Çıkışlar
    base = {
        "hist": "X_hist.npy",
        "acf": "X_acf.npy",
        "pri_track": "X_pritrack64.npy",
        "pri_track_resampled": "X_pritrack64.npy",
        "feat_vector": "X_feat_vector.npy"
    }[which_feature]

    out_X = out_dir / base
    out_y = out_dir / "y_labels.npy"
    out_meta = out_dir / "meta.json"
    out_classes = out_dir / "classes.json"
    out_ylog = out_dir / "y_logpri.npy"

    np.save(out_X, X)
    np.save(out_y, y)
    if make_logpri:
        np.save(out_ylog, y_logpri)
    with open(out_meta, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
    with open(out_classes, "w", encoding="utf-8") as fp:
        json.dump({"classes": classes_sorted, "name_to_id": name_to_id}, fp, ensure_ascii=False, indent=2)

    print(f"[OK] Saved X:       {out_X} (shape {X.shape})")
    print(f"[OK] Saved y:       {out_y} (shape {y.shape})")
    if make_logpri:
        print(f"[OK] Saved y_logpri:{out_ylog} (shape {y_logpri.shape})")
    print(f"[OK] Saved meta:    {out_meta}")
    print(f"[OK] Saved classes: {out_classes}")
    print(f"[Info] kept={kept}, skipped(low pulses)={skipped}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="Root folder with class subfolders (each has .npy)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder")
    ap.add_argument("--fs_hz", type=float, default=3_000_000)
    ap.add_argument("--nbins", type=int, default=512)
    ap.add_argument("--tmax_us", type=float, default=15000.0, help="Histogram upper bound in µs (default 15ms)")
    ap.add_argument("--smooth_len", type=int, default=9)
    ap.add_argument("--k_hi", type=float, default=4.0)
    ap.add_argument("--k_lo", type=float, default=2.0)
    ap.add_argument("--min_len_us", type=float, default=5.0)
    ap.add_argument("--min_gap_us", type=float, default=2.0)
    ap.add_argument("--include_acf", action="store_true", help="Compute ACF (auto-enabled for feat_vector)")
    ap.add_argument("--acf_nlags", type=int, default=512)
    ap.add_argument("--pri_window_pulses", type=int, default=8)
    ap.add_argument("--pri_hop_pulses", type=int, default=4)
    ap.add_argument("--pri_track_outlen", type=int, default=64)
    ap.add_argument("--which_feature", type=str, default="feat_vector", choices=FEATURE_CHOICES)
    ap.add_argument("--classes", type=str, nargs="*", default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--make_logpri", action="store_true", help="Also save y_logpri.npy for regression")
    ap.add_argument("--min_pulses", type=int, default=5, help="Min pulses for log-PRI label")
    ap.add_argument("--pri_label_json", type=str, default=None, help="Optional JSON mapping file->PRI_us")
    ap.add_argument("--logpri_eps_us", type=float, default=1e-6)
    ap.add_argument("--no_details", action="store_true", help="Do not save per-file detail npz")
    ap.add_argument("--no_tqdm", action="store_true", help="Disable progress bar")
    args = ap.parse_args()

    process_folder(
        args.in_dir, args.out_dir,
        fs_hz=args.fs_hz,
        nbins=args.nbins,
        tmax_us=args.tmax_us,
        smooth_len=args.smooth_len,
        k_hi=args.k_hi,
        k_lo=args.k_lo,
        min_len_us=args.min_len_us,
        min_gap_us=args.min_gap_us,
        include_acf=args.include_acf,
        acf_nlags=args.acf_nlags,
        pri_window_pulses=args.pri_window_pulses,
        pri_hop_pulses=args.pri_hop_pulses,
        pri_track_outlen=args.pri_track_outlen,
        which_feature=args.which_feature,
        save_details=not args.no_details,
        classes=args.classes,
        shuffle=args.shuffle,
        seed=args.seed,
        make_logpri=args.make_logpri,
        min_pulses=args.min_pulses,
        pri_label_json=args.pri_label_json,
        logpri_eps_us=args.logpri_eps_us,
        no_tqdm=args.no_tqdm,
    )

if __name__ == "__main__":
    main()
