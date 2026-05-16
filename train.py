import os
import json
import argparse
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Hungarian
from scipy.optimize import linear_sum_assignment

from modules import UNet1D
from tqdm import tqdm  # progress bar

# ----------------------
# Utils
# ----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def auto_device(pref: str):
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(pref)

# ----------------------
# Dataset
# ----------------------
class WindowedRadarDataset(Dataset):
    """
    X: (B, 2, L), Y: (B, N, L)
    Pencereleme ile (frame, start, end) alt-dilimlerini üretir ve bunları
    GRU'nun eğitimi için 'seq_len' uzunluğunda ardışık diziler (sequences) haline getirir.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 use_windows: bool, window_len: int, window_stride: int,
                 include_tail_window: bool, seq_len: int = 4): # GÜNCELLEME: seq_len eklendi
        assert X.ndim == 3 and Y.ndim == 3, "X and Y must be (B, C/E, L)"
        assert X.shape[0] == Y.shape[0] and X.shape[2] == Y.shape[2], "Batch and length must match"
        self.X = X
        self.Y = Y
        self.use_windows = use_windows
        self.window_len = window_len
        self.window_stride = window_stride
        self.seq_len = seq_len

        self.sequences = []  # Artık tekil pencereler değil, pencere listeleri tutuyoruz
        B, _, L = X.shape
        
        if not use_windows:
            for b in range(B):
                # Eğer pencereleme yoksa, tek pencereyi seq_len=1 olarak sarıyoruz
                self.sequences.append([(b, 0, L)])
        else:
            for b in range(B):
                b_windows = []
                start = 0
                while start + window_len <= L:
                    b_windows.append((b, start, start + window_len))
                    start += window_stride
                
                # Pencereleri seq_len uzunluğunda parçalara (chunk) ayır
                # Drop_last mantığı kullanıyoruz ki tensör boyutları sabit kalsın
                for i in range(0, len(b_windows) - seq_len + 1, seq_len):
                    self.sequences.append(b_windows[i : i + seq_len])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x_seq, y_seq = [], []
        
        # Dizideki her bir pencereyi çıkarıp biriktiriyoruz
        for (b, s, e) in seq:
            x_seq.append(self.X[b, :, s:e])      # (2, T)
            y_seq.append(self.Y[b, :, s:e])      # (N, T)
            
        x_seq = np.stack(x_seq, axis=0) # (seq_len, 2, T)
        y_seq = np.stack(y_seq, axis=0) # (seq_len, N, T)
        
        x_tsr = torch.from_numpy(x_seq).float()
        y_tsr = torch.from_numpy(y_seq).float()
        return x_tsr, y_tsr

# ----------------------
# Losses & PIT
# ----------------------
def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    logits:  (E, T)
    targets: (E, T)  (assignment sonrası hizalı)
    """
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=-1)
    den = probs.sum(dim=-1) + targets.sum(dim=-1) + eps
    dice = 1.0 - (num / den)  # (E,)
    return dice.mean()

def compute_pulse_center_and_duration(mask: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Darbenin merkez noktasını ve süresini hesaplar.
    mask: (T,) binary mask veya probability
    returns: (center, duration)
    """
    binary = (mask >= threshold).float()
    active_indices = torch.nonzero(binary, as_tuple=False).flatten()
    
    if active_indices.numel() == 0:
        return torch.tensor(0.0, device=mask.device), torch.tensor(0.0, device=mask.device)
    
    start = active_indices[0].float()
    end = active_indices[-1].float()
    center = (start + end) / 2.0
    duration = (end - start + 1.0)
    
    return center, duration

def compute_pulse_start_end(mask: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Darbenin başlangıç ve bitiş noktalarını hesaplar.
    mask: (T,) binary mask veya probability
    returns: (start, end)
    """
    binary = (mask >= threshold).float()
    active_indices = torch.nonzero(binary, as_tuple=False).flatten()
    
    if active_indices.numel() == 0:
        return torch.tensor(0.0, device=mask.device), torch.tensor(0.0, device=mask.device)
    
    start = active_indices[0].float()
    end = active_indices[-1].float()
    
    return start, end

def compute_position_cost(pred_logits: torch.Tensor, gt_masks: torch.Tensor,
                          threshold: float = 0.5, 
                          center_weight: float = 0.1,
                          duration_weight: float = 0.1,
                          start_weight: float = 0.05,
                          end_weight: float = 0.05,
                          max_time: float = None,
                          use_relative_cost: bool = True) -> torch.Tensor:
    """
    İyileştirilmiş konum benzerliği maliyeti:
    - Merkez farkı
    - Süre farkı
    - Başlangıç farkı
    - Bitiş farkı
    
    pred_logits: (E, T)
    gt_masks:    (E_true, T)
    returns: (E, E_true) position cost matrix
    """
    E, T = pred_logits.shape
    Egt = gt_masks.shape[0]
    
    if max_time is None:
        max_time = float(T)
    
    pred_probs = torch.sigmoid(pred_logits)
    
    # Her pred ve GT için tüm konum özelliklerini hesapla
    pred_centers = []
    pred_durations = []
    pred_starts = []
    pred_ends = []
    for i in range(E):
        c, d = compute_pulse_center_and_duration(pred_probs[i], threshold)
        s, e = compute_pulse_start_end(pred_probs[i], threshold)
        pred_centers.append(c)
        pred_durations.append(d)
        pred_starts.append(s)
        pred_ends.append(e)
    
    gt_centers = []
    gt_durations = []
    gt_starts = []
    gt_ends = []
    for j in range(Egt):
        c, d = compute_pulse_center_and_duration(gt_masks[j], threshold)
        s, e = compute_pulse_start_end(gt_masks[j], threshold)
        gt_centers.append(c)
        gt_durations.append(d)
        gt_starts.append(s)
        gt_ends.append(e)
    
    pred_centers = torch.stack(pred_centers)  # (E,)
    pred_durations = torch.stack(pred_durations)  # (E,)
    pred_starts = torch.stack(pred_starts)  # (E,)
    pred_ends = torch.stack(pred_ends)  # (E,)
    gt_centers = torch.stack(gt_centers)  # (E_true,)
    gt_durations = torch.stack(gt_durations)  # (E_true,)
    gt_starts = torch.stack(gt_starts)  # (E_true,)
    gt_ends = torch.stack(gt_ends)  # (E_true,)
    
    # Pairwise farklar
    center_diff = (pred_centers.unsqueeze(1) - gt_centers.unsqueeze(0)).abs()  # (E, E_true)
    duration_diff = (pred_durations.unsqueeze(1) - gt_durations.unsqueeze(0)).abs()  # (E, E_true)
    start_diff = (pred_starts.unsqueeze(1) - gt_starts.unsqueeze(0)).abs()  # (E, E_true)
    end_diff = (pred_ends.unsqueeze(1) - gt_ends.unsqueeze(0)).abs()  # (E, E_true)
    
    if use_relative_cost:
        # Relatif maliyet: farkları normalize et ve daha agresif hale getir
        # Merkez farkı: normalize edilmiş ve kare alınmış (büyük farkları cezalandır)
        center_cost = center_weight * torch.pow(center_diff / max_time, 2.0)
        # Süre farkı: normalize edilmiş ve kare alınmış
        avg_duration = (pred_durations.mean() + gt_durations.mean()) / 2.0 + 1e-6
        duration_cost = duration_weight * torch.pow(duration_diff / avg_duration, 2.0)
        # Başlangıç ve bitiş farkları
        start_cost = start_weight * torch.pow(start_diff / max_time, 2.0)
        end_cost = end_weight * torch.pow(end_diff / max_time, 2.0)
    else:
        # Basit normalize edilmiş maliyet
        center_cost = center_weight * (center_diff / max_time)
        duration_cost = duration_weight * (duration_diff / (max_time + 1e-6))
        start_cost = start_weight * (start_diff / max_time)
        end_cost = end_weight * (end_diff / max_time)
    
    position_cost = center_cost + duration_cost + start_cost + end_cost
    return position_cost

def pairwise_cost_matrix(pred_logits: torch.Tensor, gt_masks: torch.Tensor,
                         lambda_bce: float = 0.6, 
                         bce_pos_weight: float = 1.0,
                         use_dynamic_bce_pos_weight: bool = True,
                         use_spatial_cost: bool = True,
                         spatial_cost_weight: float = 0.15,
                         center_weight: float = 0.1,
                         duration_weight: float = 0.1,
                         start_weight: float = 0.05,
                         end_weight: float = 0.05,
                         use_relative_spatial_cost: bool = True) -> torch.Tensor:
    """
    İyileştirilmiş maliyet matrisi:
    - Dinamik BCE pos_weight: Her GT maskesi için pozitif oranına göre (class imbalance için)
    - Konum benzerliği maliyeti: Merkez ve süre farkı (spatial matching için)
    
    Args:
        pred_logits: (E, T) - Prediction logits
        gt_masks: (E_true, T) - Ground truth masks (sadece boş olmayan GT kanallar)
        lambda_bce: BCE ve Dice arasındaki ağırlık (0-1)
        bce_pos_weight: BCE loss için pozitif sınıf ağırlığı (class imbalance)
        use_dynamic_bce_pos_weight: Dinamik BCE pos_weight kullan (kısa darbeler için daha yüksek)
        use_spatial_cost: Konum benzerliği maliyeti ekle
        spatial_cost_weight: Konum maliyetinin genel ağırlığı (0-1 arası önerilir)
        center_weight: Merkez farkının ağırlığı
        duration_weight: Süre farkının ağırlığı
    
    Returns:
        cost: (E, E_true) cost matrix for Hungarian algorithm
    """
    E, T = pred_logits.shape
    assert gt_masks.shape[1] == T, "time length mismatch"
    Egt = gt_masks.shape[0]

    pred = pred_logits.unsqueeze(1).expand(E, Egt, T)  # (E, Egt, T)
    gt   = gt_masks.unsqueeze(0).expand(E, Egt, T)     # (E, Egt, T)

    # Parametre validasyonu
    assert 0.0 <= lambda_bce <= 1.0, f"lambda_bce must be in [0, 1], got {lambda_bce}"
    assert bce_pos_weight >= 1.0, f"bce_pos_weight must be >= 1.0, got {bce_pos_weight}"
    assert 0.0 <= spatial_cost_weight <= 1.0, f"spatial_cost_weight should be in [0, 1], got {spatial_cost_weight}"
    
    # BCE - Dinamik BCE pos_weight ile (class imbalance için)
    if use_dynamic_bce_pos_weight and bce_pos_weight > 1.0:
        # Her GT maskesi için pozitif oranına göre dinamik BCE pos_weight
        gt_positive_ratios = gt_masks.mean(dim=-1)  # (E_true,) - her GT için pozitif oranı
        # Kısa darbeler (düşük pozitif oran) için daha yüksek pos_weight
        # Yumuşak formül: base_pos_weight * sqrt(1 / (positive_ratio + eps))
        # Bu, çok yüksek değerlere çıkmayı önler
        eps = 1e-6
        # Karekök kullanarak daha yumuşak bir scaling
        ratio_factor = torch.sqrt(1.0 / (gt_positive_ratios + eps))  # (E_true,)
        dynamic_pw = bce_pos_weight * ratio_factor  # (E_true,)
        # Clamp to reasonable range: maksimum bce_pos_weight'ın 3 katı (25 -> 75 max)
        max_pw = min(bce_pos_weight * 3.0, 100.0)  # Çok yüksek değerleri önle
        dynamic_pw = torch.clamp(dynamic_pw, min=1.0, max=max_pw)
        # Expand to (E, Egt) for broadcasting
        dynamic_pw = dynamic_pw.unsqueeze(0).expand(E, Egt)  # (E, Egt)
        # Reshape for BCE
        dynamic_pw = dynamic_pw.unsqueeze(-1).expand(E, Egt, T)  # (E, Egt, T)
        bce = F.binary_cross_entropy_with_logits(pred, gt, pos_weight=dynamic_pw, reduction="none").mean(dim=-1)
    elif bce_pos_weight > 1.0:
        pw = torch.full((1,), fill_value=bce_pos_weight, device=pred_logits.device)
        bce = F.binary_cross_entropy_with_logits(pred, gt, pos_weight=pw, reduction="none").mean(dim=-1)
    else:
        bce = F.binary_cross_entropy_with_logits(pred, gt, reduction="none").mean(dim=-1)

    # Dice benzeri terim
    probs = torch.sigmoid(pred)
    inter = 2.0 * (probs * gt).sum(dim=-1)
    union = probs.sum(dim=-1) + gt.sum(dim=-1) + 1e-6
    dice = 1.0 - (inter / union)

    # Temel maliyet: BCE + Dice (lambda_bce düşürülerek Dice'a daha fazla ağırlık verilir)
    base_cost = lambda_bce * bce + (1.0 - lambda_bce) * dice
    
    # Konum benzerliği maliyeti ekle (spatial matching için)
    if use_spatial_cost:
        spatial_cost = compute_position_cost(
            pred_logits, gt_masks,
            threshold=0.5,
            center_weight=center_weight,
            duration_weight=duration_weight,
            start_weight=start_weight,
            end_weight=end_weight,
            max_time=float(T),
            use_relative_cost=use_relative_spatial_cost
        )
        # Konum maliyetini normalize et ve ekle
        # Spatial cost'u base_cost ile aynı ölçekte olacak şekilde normalize et
        base_cost_mean = base_cost.mean()
        spatial_cost_mean = spatial_cost.mean() + 1e-6
        # Spatial cost'u base_cost'un ortalama değerine göre normalize et
        spatial_cost_scaled = spatial_cost * (base_cost_mean / spatial_cost_mean)
        # Kombine maliyet: base_cost + ağırlıklandırılmış spatial_cost
        cost = base_cost + spatial_cost_weight * spatial_cost_scaled
    else:
        cost = base_cost
    
    return cost

def pit_loss_for_sample(pred_logits: torch.Tensor,
                        gt_masks: torch.Tensor,
                        lambda_bce: float,
                        empty_weight: float,
                        bce_pos_weight: float,
                        exist_thr: float = None,
                        use_dynamic_bce_pos_weight: bool = True,
                        use_spatial_cost: bool = True,
                        spatial_cost_weight: float = 0.15,
                        center_weight: float = 0.1,
                        duration_weight: float = 0.1,
                        start_weight: float = 0.05,
                        end_weight: float = 0.05,
                        use_relative_spatial_cost: bool = True) -> torch.Tensor:
    """
    pred_logits: (N, T)
    gt_masks:    (N, T)   # bazı GT kanallar tamamen 0 olabilir
    """
    N, T = pred_logits.shape

    # 1) Boş olmayan GT kanallar
    gt_sums = gt_masks.sum(dim=-1)                 # (N,)
    nonempty_idx = torch.nonzero(gt_sums > 0, as_tuple=False).flatten()

    loss = 0.0
    matched_pred = set()
    candidate_idx = torch.arange(N, device=pred_logits.device)
    if exist_thr is not None:
        max_probs = torch.sigmoid(pred_logits).amax(dim=-1)
        candidate_idx = torch.nonzero(max_probs >= exist_thr, as_tuple=False).flatten()
        if candidate_idx.numel() == 0:
            candidate_idx = torch.argmax(max_probs).unsqueeze(0)
    candidate_idx_list = candidate_idx.tolist()

    if nonempty_idx.numel() > 0:
        # Hungarian sadece boş olmayan GT üzerinde
        cost_mat = pairwise_cost_matrix(
            pred_logits[candidate_idx], gt_masks[nonempty_idx],
            lambda_bce=lambda_bce, 
            bce_pos_weight=bce_pos_weight,
            use_dynamic_bce_pos_weight=use_dynamic_bce_pos_weight,
            use_spatial_cost=use_spatial_cost,
            spatial_cost_weight=spatial_cost_weight,
            center_weight=center_weight,
            duration_weight=duration_weight,
            start_weight=start_weight,
            end_weight=end_weight,
            use_relative_spatial_cost=use_relative_spatial_cost
        )
        row_ind, col_ind = linear_sum_assignment(cost_mat.detach().cpu().numpy())

        # Eşleşen çiftler (N_pred_i ↔ GT_j)
        for i, j_local in zip(row_ind, col_ind):
            j = nonempty_idx[j_local].item()
            real_i = candidate_idx_list[i]
            p = pred_logits[real_i]
            g = gt_masks[j]
            if bce_pos_weight > 1.0:
                bce = F.binary_cross_entropy_with_logits(p, g, pos_weight=torch.tensor(bce_pos_weight, device=p.device))
            else:
                bce = F.binary_cross_entropy_with_logits(p, g)
            d = dice_loss_from_logits(p.unsqueeze(0), g.unsqueeze(0))
            pair_loss = lambda_bce * bce + (1.0 - lambda_bce) * d
            loss += pair_loss
            matched_pred.add(int(real_i))

        # Normalize: boş olmayan GT sayısına göre
        loss = loss / max(1, nonempty_idx.numel())

    # 2) Eşleşmemiş pred kanallarına "boş hedef"e karşı down-weighted ceza
    if len(matched_pred) < N:
        zeros = torch.zeros(T, device=pred_logits.device)
        for i in range(N):
            if i not in matched_pred:
                loss += empty_weight * F.binary_cross_entropy_with_logits(pred_logits[i], zeros)

    return loss

def pit_loss_batch(pred_logits_b: torch.Tensor,
                   gt_masks_b: torch.Tensor,
                   lambda_bce: float,
                   empty_weight: float,
                   bce_pos_weight: float,
                   exist_thr: float = None,
                   use_dynamic_bce_pos_weight: bool = True,
                   use_spatial_cost: bool = True,
                        spatial_cost_weight: float = 0.15,
                        center_weight: float = 0.1,
                        duration_weight: float = 0.1,
                        start_weight: float = 0.05,
                        end_weight: float = 0.05,
                        use_relative_spatial_cost: bool = True) -> torch.Tensor:
    """
    Batch için PIT loss hesaplar.
    
    Args:
        pred_logits_b: (B, N, T) - Batch prediction logits
        gt_masks_b: (B, N, T) - Batch ground truth masks
        lambda_bce: BCE ve Dice arasındaki ağırlık
        empty_weight: Eşleşmemiş pred kanalları için boş hedef cezası
        bce_pos_weight: BCE loss için pozitif sınıf ağırlığı
        exist_thr: Pred kanalının "var" kabul edilmesi için maksimum prob eşiği
        use_dynamic_bce_pos_weight: Dinamik BCE pos_weight kullan
        use_spatial_cost: Konum benzerliği maliyeti kullan
        spatial_cost_weight: Konum maliyetinin ağırlığı
        center_weight: Merkez farkının ağırlığı
        duration_weight: Süre farkının ağırlığı
    
    Returns:
        loss: Ortalama PIT loss
    """
    B = pred_logits_b.shape[0]
    total = 0.0
    for b in range(B):
        total += pit_loss_for_sample(
            pred_logits_b[b], gt_masks_b[b],
            lambda_bce=lambda_bce,
            empty_weight=empty_weight,
            bce_pos_weight=bce_pos_weight,
            exist_thr=exist_thr,
            use_dynamic_bce_pos_weight=use_dynamic_bce_pos_weight,
            use_spatial_cost=use_spatial_cost,
            spatial_cost_weight=spatial_cost_weight,
            center_weight=center_weight,
            duration_weight=duration_weight,
            start_weight=start_weight,
            end_weight=end_weight,
            use_relative_spatial_cost=use_relative_spatial_cost
        )
    return total / B

# ----------------------
# Validation metrics (PIT'li)
# ----------------------
@torch.no_grad()
def _soft_dice(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    probs:   (T,)  in [0,1]
    targets: (T,)  in {0,1}
    """
    inter = 2.0 * (probs * targets).sum()
    union = probs.sum() + targets.sum() + eps
    return inter / union

@torch.no_grad()
def _f1_from_threshold(pred_bin: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    """
    pred_bin: (T,) in {0,1}
    targets:  (T,) in {0,1}
    returns (tp, fp, fn, f1)
    """
    tp = (pred_bin & (targets == 1)).sum().float()
    fp = (pred_bin & (targets == 0)).sum().float()
    fn = ((pred_bin == 0) & (targets == 1)).sum().float()
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return tp, fp, fn, f1

@torch.no_grad()
def evaluate_pit_metrics_batch(logits_b: torch.Tensor,
                               gt_masks_b: torch.Tensor,
                               thr: float = 0.5,
                               exist_thr: float = 0.35):
    """
    PIT ile eşleştirerek:
      - mean soft Dice (eşleşen çiftlerde)
      - mikro F1 (eşleşmeyen pred kanallarının pozitifleri FP sayılır)
    döndürür.
    logits_b: (B, N, T)
    gt_masks_b: (B, N, T)
    """
    B, N, T = logits_b.shape
    probs_b = torch.sigmoid(logits_b)

    dice_sum = 0.0
    dice_cnt = 0
    tp_tot = torch.tensor(0.0, device=logits_b.device)
    fp_tot = torch.tensor(0.0, device=logits_b.device)
    fn_tot = torch.tensor(0.0, device=logits_b.device)

    for b in range(B):
        probs = probs_b[b]           # (N, T)
        gt = gt_masks_b[b]           # (N, T)
        gt_sums = gt.sum(dim=-1)     # (N,)
        nonempty_idx = torch.nonzero(gt_sums > 0, as_tuple=False).flatten()

        matched_pred = set()
        cand_idx = torch.arange(N, device=logits_b.device)
        if exist_thr is not None:
            max_probs = probs.amax(dim=-1)
            cand_idx = torch.nonzero(max_probs >= exist_thr, as_tuple=False).flatten()
            if cand_idx.numel() == 0:
                cand_idx = torch.argmax(max_probs).unsqueeze(0)
        cand_idx_list = cand_idx.tolist()
        if nonempty_idx.numel() > 0:
            # Eşleştirme: soft-dice'a göre (cost = 1 - dice)
            Egt = nonempty_idx.numel()
            # pairwise soft-dice
            # (N, Egt) matris:
            P = probs[cand_idx].unsqueeze(1).expand(cand_idx.shape[0], Egt, T)
            G = gt[nonempty_idx].unsqueeze(0).expand(P.shape[0], Egt, T)
            inter = 2.0 * (P * G).sum(dim=-1)
            union = P.sum(dim=-1) + G.sum(dim=-1) + 1e-6
            dice_pair = inter / union                      # (N, Egt)
            cost = (1.0 - dice_pair).detach().cpu().numpy()

            row_ind, col_ind = linear_sum_assignment(cost)

            # Eşleşen çiftlerde Dice & F1
            for i, j_local in zip(row_ind, col_ind):
                j = nonempty_idx[j_local].item()
                real_i = cand_idx_list[i]
                p = probs[real_i]              # (T,)
                g = gt[j].bool()          # (T,)
                dice = _soft_dice(p, g.float())
                dice_sum += dice.item()
                dice_cnt += 1

                pred_bin = (p >= thr)
                tp, fp, fn, _ = _f1_from_threshold(pred_bin, g)
                tp_tot += tp; fp_tot += fp; fn_tot += fn
                matched_pred.add(int(real_i))

            # Teorik olarak tüm non-empty GT'ler eşleşti (Hungarian), ama kontrol:
            # Eşleşmeyen non-empty GT varsa, onlar tamamen FN sayılır:
            matched_gt_local = set(col_ind.tolist())
            for j_local in range(Egt):
                if j_local not in matched_gt_local:
                    g = gt[nonempty_idx[j_local]].bool()
                    # tüm pozitifler FN
                    fn_tot += g.sum()

        # Eşleşmeyen pred kanalları: "exist" ise FP
        if len(matched_pred) < N:
            for i in range(N):
                if i in matched_pred:
                    continue
                p = probs[i]
                if exist_thr is None or p.max() >= exist_thr:
                    pred_bin = (p >= thr)
                    fp_tot += pred_bin.sum()

    # Ortalama soft Dice (sadece matched çiftlerde)
    mean_dice = (dice_sum / max(1, dice_cnt))
    f1 = (2 * tp_tot) / (2 * tp_tot + fp_tot + fn_tot + 1e-6)
    return float(mean_dice), float(f1.item())

def sequence_pit_loss_for_sample(pred_logits_seq: torch.Tensor,
                                 gt_masks_seq: torch.Tensor,
                                 # ... [Diğer parametreler aynı: lambda_bce, empty_weight, vb.] ...
                                 lambda_bce: float, empty_weight: float, bce_pos_weight: float,
                                 use_dynamic_bce_pos_weight: bool = True, use_spatial_cost: bool = True,
                                 spatial_cost_weight: float = 0.15, center_weight: float = 0.1,
                                 duration_weight: float = 0.1, start_weight: float = 0.05,
                                 end_weight: float = 0.05, use_relative_spatial_cost: bool = True):
    """
    Ardışık pencereler (sequence) üzerinden tek bir kimlik (kanal) eşleştirmesi yapar.
    pred_logits_seq: (S, N, T) -> S: seq_len
    gt_masks_seq:    (S, N, T) 
    """
    S, N, T = pred_logits_seq.shape
    
    # Tüm dizi boyunca aktif olan GT kanallarını bul
    gt_sums_seq = gt_masks_seq.sum(dim=(0, 2))  # (N,) - S ve T boyutlarında topla
    nonempty_idx = torch.nonzero(gt_sums_seq > 0, as_tuple=False).flatten()

    loss = 0.0
    matched_pred = set()
    candidate_idx = torch.arange(N, device=pred_logits_seq.device)
    candidate_idx_list = candidate_idx.tolist()

    if nonempty_idx.numel() > 0:
        E_gt = nonempty_idx.numel()
        # Tüm dizi (sequence) için kümülatif maliyet matrisi
        total_cost_mat = torch.zeros((N, E_gt), device=pred_logits_seq.device)
        
        # Her bir zaman adımı (pencere) için cost hesaplayıp topluyoruz
        for s in range(S):
            # Sadece o adımda boş da olsa, GT ile Pred arasındaki farkı ekliyoruz
            step_cost = pairwise_cost_matrix(
                pred_logits_seq[s, candidate_idx], gt_masks_seq[s, nonempty_idx],
                lambda_bce=lambda_bce, bce_pos_weight=bce_pos_weight,
                use_dynamic_bce_pos_weight=use_dynamic_bce_pos_weight,
                use_spatial_cost=use_spatial_cost, spatial_cost_weight=spatial_cost_weight,
                center_weight=center_weight, duration_weight=duration_weight,
                start_weight=start_weight, end_weight=end_weight,
                use_relative_spatial_cost=use_relative_spatial_cost
            )
            total_cost_mat += step_cost
            
        # Kümülatif maliyet üzerinden Hungarian ataması (Tüm Sequence için TEK BİR ATAMA)
        row_ind, col_ind = linear_sum_assignment(total_cost_mat.detach().cpu().numpy())

        # Atanan kanallara göre tüm dizi boyunca loss hesapla
        for s in range(S):
            step_loss = 0.0
            for i, j_local in zip(row_ind, col_ind):
                j = nonempty_idx[j_local].item()
                real_i = candidate_idx_list[i]
                
                p = pred_logits_seq[s, real_i]
                g = gt_masks_seq[s, j]
                
                if bce_pos_weight > 1.0:
                    bce = F.binary_cross_entropy_with_logits(p, g, pos_weight=torch.tensor(bce_pos_weight, device=p.device))
                else:
                    bce = F.binary_cross_entropy_with_logits(p, g)
                    
                d = dice_loss_from_logits(p.unsqueeze(0), g.unsqueeze(0))
                step_loss += lambda_bce * bce + (1.0 - lambda_bce) * d
                
                if s == 0: # Set'e sadece ilk adımda eklemek yeterli
                    matched_pred.add(int(real_i))
                    
            loss += step_loss / max(1, E_gt)
            
        loss = loss / S # Dizi uzunluğuna bölüp ortalama alıyoruz

    # Eşleşmeyen pred kanallarına "boş hedef" cezası (Tüm S adımları için)
    if len(matched_pred) < N:
        zeros = torch.zeros(T, device=pred_logits_seq.device)
        empty_loss = 0.0
        for s in range(S):
            for i in range(N):
                if i not in matched_pred:
                    empty_loss += F.binary_cross_entropy_with_logits(pred_logits_seq[s, i], zeros)
        loss += empty_weight * (empty_loss / S)

    return loss

def sequence_pit_loss_batch(pred_logits_seq_b: torch.Tensor,
                            gt_masks_seq_b: torch.Tensor,
                            # ... [Parametreler üsttekiyle aynı kalacak] ...
                            lambda_bce: float, empty_weight: float, bce_pos_weight: float,
                            use_dynamic_bce_pos_weight: bool = True, use_spatial_cost: bool = True,
                            spatial_cost_weight: float = 0.15, center_weight: float = 0.1,
                            duration_weight: float = 0.1, start_weight: float = 0.05,
                            end_weight: float = 0.05, use_relative_spatial_cost: bool = True):
    """
    Batch için Sequence PIT loss hesaplar.
    pred_logits_seq_b: (B, S, N, T)
    gt_masks_seq_b:    (B, S, N, T)
    """
    B = pred_logits_seq_b.shape[0]
    total = 0.0
    for b in range(B):
        total += sequence_pit_loss_for_sample(
            pred_logits_seq_b[b], gt_masks_seq_b[b],
            lambda_bce=lambda_bce, empty_weight=empty_weight, bce_pos_weight=bce_pos_weight,
            use_dynamic_bce_pos_weight=use_dynamic_bce_pos_weight, use_spatial_cost=use_spatial_cost,
            spatial_cost_weight=spatial_cost_weight, center_weight=center_weight,
            duration_weight=duration_weight, start_weight=start_weight, end_weight=end_weight,
            use_relative_spatial_cost=use_relative_spatial_cost
        )
    return total / max(1, B)

def run_training(cfg_path: str):
    cfg = load_config(cfg_path)
    set_seed(cfg["training"].get("seed", 42))

    device = auto_device(cfg.get("device", "auto"))
    print(f"[Device] {device}")

    # Veri
    X = np.load(cfg["data"]["train_data_path"])
    Y = np.load(cfg["data"]["train_labels_path"])
    assert X.ndim == 3 and Y.ndim == 3, "Expect X:(B,2,L), Y:(B,N,L)"
    B, Cin, L = X.shape
    N = Y.shape[1]
    assert Cin == cfg["model"]["in_channels"]
    assert N == cfg["model"]["num_emitters"]
    
    # Veri normalizasyonu - Her kanal için ayrı ayrı
    print(f"[Data] Original: mean={X.mean():.4f}, std={X.std():.4f}")
    for ch in range(Cin):
        ch_mean = X[:, ch, :].mean()
        ch_std = X[:, ch, :].std()
        X[:, ch, :] = (X[:, ch, :] - ch_mean) / (ch_std + 1e-8)
    print(f"[Data] Normalized: mean={X.mean():.4f}, std={X.std():.4f}")
    
    # Label statistics
    pos_ratio = np.mean(Y > 0.5)
    print(f"[Labels] Positive ratio: {pos_ratio:.6f}")
    print(f"[Labels] Data shape: {X.shape}, Labels shape: {Y.shape}")

    # Train/val split
    val_split = float(cfg["data"].get("val_split", 0.2))
    indices = np.arange(B)
    if cfg["data"].get("shuffle", True):
        np.random.shuffle(indices)
    val_count = max(1, int(round(B * val_split)))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    if len(train_idx) == 0:
        train_idx, val_idx = indices[::2], indices[1::2]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val     = X[val_idx], Y[val_idx]
    
    print(f"[Data] Validation samples: {len(val_idx)}, Training samples: {len(train_idx)}")

    # Datasets & Loaders
    wcfg = cfg["windowing"]
    ds_train = WindowedRadarDataset(
        X_train, Y_train,
        use_windows=wcfg["use_windows"],
        window_len=wcfg["window_len"],
        window_stride=wcfg["window_stride"],
        include_tail_window=wcfg.get("include_tail_window", True),
    )
    ds_val = WindowedRadarDataset(
        X_val, Y_val,
        use_windows=wcfg["use_windows"],
        window_len=wcfg["window_len"],
        window_stride=wcfg["window_stride"],
        include_tail_window=wcfg.get("include_tail_window", True),
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    mcfg = cfg["model"]
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
    ).to(device)

    # Optimizasyon
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg["training"].get("amp", True) and device.type == "cuda"
    )
    epochs = int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-5
    )

    lambda_bce = float(cfg["training"]["lambda_bce"])
    bce_pos_weight = float(cfg["training"].get("bce_pos_weight", cfg["training"].get("pos_weight", 1.0)))  # Backward compatibility
    empty_weight = float(cfg["training"]["empty_weight"])
    grad_clip = float(cfg["training"]["grad_clip"])
    exist_thr = float(cfg["training"].get("exist_threshold", 0.35))
    
    # İyileştirilmiş maliyet fonksiyonu parametreleri
    use_dynamic_bce_pos_weight = cfg["training"].get("use_dynamic_bce_pos_weight", 
                                                      cfg["training"].get("use_dynamic_pos_weight", True))  # Backward compatibility
    use_spatial_cost = cfg["training"].get("use_spatial_cost", 
                                          cfg["training"].get("use_position_cost", True))  # Backward compatibility
    spatial_cost_weight = float(cfg["training"].get("spatial_cost_weight", 
                                                    cfg["training"].get("position_weight", 0.15)))  # Backward compatibility
    center_weight = float(cfg["training"].get("center_weight", 0.1))
    duration_weight = float(cfg["training"].get("duration_weight", 0.1))
    start_weight = float(cfg["training"].get("start_weight", 0.05))
    end_weight = float(cfg["training"].get("end_weight", 0.05))
    use_relative_spatial_cost = cfg["training"].get("use_relative_spatial_cost", True)
    
    # Parametre validasyonu ve uyarılar
    if "pos_weight" in cfg["training"] and "bce_pos_weight" not in cfg["training"]:
        print(f"[Warning] 'pos_weight' is deprecated. Use 'bce_pos_weight' instead. Using {bce_pos_weight}.")
    if "use_dynamic_pos_weight" in cfg["training"] and "use_dynamic_bce_pos_weight" not in cfg["training"]:
        print(f"[Warning] 'use_dynamic_pos_weight' is deprecated. Use 'use_dynamic_bce_pos_weight' instead.")
    if "use_position_cost" in cfg["training"] and "use_spatial_cost" not in cfg["training"]:
        print(f"[Warning] 'use_position_cost' is deprecated. Use 'use_spatial_cost' instead.")
    if "position_weight" in cfg["training"] and "spatial_cost_weight" not in cfg["training"]:
        print(f"[Warning] 'position_weight' is deprecated. Use 'spatial_cost_weight' instead.")

    out_dir = cfg["save"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    model_name = cfg["save"]["model_name"]
    
    # Validation verisini kaydet (test için) - TÜM windowed validation verisi
    val_data_path = os.path.join(out_dir, "val_data.npz")
    
    # Windowed validation dataset oluştur (test ile aynı windowing)
    wcfg = cfg["windowing"]
    ds_val_for_save = WindowedRadarDataset(
        X_val, Y_val,
        use_windows=wcfg["use_windows"],
        window_len=wcfg["window_len"],
        window_stride=wcfg["window_stride"],
        include_tail_window=wcfg.get("include_tail_window", True),
    )
    
    print(f"[Data] Creating windowed validation dataset...")
    print(f"[Data] Total validation windows: {len(ds_val_for_save)}")
    
    # TÜM validation window'larını kaydet
    X_val_windows = []
    Y_val_windows = []
    
    print(f"[Data] Saving all {len(ds_val_for_save)} validation windows...")
    for i in range(len(ds_val_for_save)):
        if i % 100 == 0:  # Progress indicator
            print(f"[Data] Processing window {i+1}/{len(ds_val_for_save)}")
        x, y = ds_val_for_save[i]
        X_val_windows.append(x.numpy())
        Y_val_windows.append(y.numpy())
    
    X_val_windows = np.stack(X_val_windows, axis=0)  # (num_windows, 2, window_len)
    Y_val_windows = np.stack(Y_val_windows, axis=0)  # (num_windows, num_emitters, window_len)
    
    # Windowing bilgilerini de kaydet
    windowing_info = {
        "use_windows": wcfg["use_windows"],
        "window_len": wcfg["window_len"],
        "window_stride": wcfg["window_stride"],
        "include_tail_window": wcfg.get("include_tail_window", True),
        "total_windows": len(ds_val_for_save)
    }
    
    np.savez(val_data_path, 
             X_val=X_val_windows, Y_val=Y_val_windows, 
             val_indices=val_idx, train_indices=train_idx,
             data_config={"val_split": val_split, "shuffle": cfg["data"].get("shuffle", True)},
             windowing_config=windowing_info)
    print(f"[Data] Validation data saved to: {val_data_path}")
    print(f"[Data] Saved ALL {len(ds_val_for_save)} windows for testing (shape: {X_val_windows.shape})")
    print(f"[Data] Windowing config: {windowing_info}")

    best_val = float("inf")
    best_report = None
    
    # Debug için loss history
    train_losses = []
    val_losses = []
    val_dices = []
    val_f1s = []

    for ep in range(1, epochs + 1):
        print(f"\nEpoch {ep}/{epochs}  (lr={opt.param_groups[0]['lr']:.2e})")

        # --------- Train ---------
        # --------- Train ---------
        model.train()
        tot_loss, n_train = 0.0, 0
        pbar = tqdm(dl_train, desc=f"Train [{len(ds_train)} seqs]", leave=False)
        for xb_seq, yb_seq in pbar:
            # xb_seq: (B, S, 2, T), yb_seq: (B, S, N, T)
            xb_seq = xb_seq.to(device)
            yb_seq = yb_seq.to(device)
            
            B, S, _, T = xb_seq.shape
            N = yb_seq.shape[2]

            opt.zero_grad(set_to_none=True)
            
            # Dizinin (sequence) tüm çıktılarını biriktireceğimiz liste
            logits_seq_list = []
            
            # Her yeni dizi (batch) başladığında hafızayı (h) sıfırlıyoruz (None yapıyoruz)
            h = None 
            
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                # Dizi (Sequence) boyunca pencereleri sırayla modele ver
                for s in range(S):
                    # s. pencereyi al: (B, 2, T)
                    xb_step = xb_seq[:, s, :, :] 
                    
                    # Modeli h (hafıza) ile çağır ve yeni h'yi al
                    logits_step, h = model(xb_step, h) 
                    logits_seq_list.append(logits_step)
                
                # Biriken tahminleri tek tensörde birleştir: (B, S, N, T)
                logits_seq_tensor = torch.stack(logits_seq_list, dim=1)
                
                # Yeni Sequence PIT Loss fonksiyonumuzu çağırıyoruz
                loss = sequence_pit_loss_batch(
                    logits_seq_tensor, yb_seq,
                    lambda_bce=lambda_bce,
                    empty_weight=empty_weight,
                    bce_pos_weight=bce_pos_weight,
                    use_dynamic_bce_pos_weight=use_dynamic_bce_pos_weight,
                    use_spatial_cost=use_spatial_cost,
                    spatial_cost_weight=spatial_cost_weight,
                    center_weight=center_weight,
                    duration_weight=duration_weight,
                    start_weight=start_weight,
                    end_weight=end_weight,
                    use_relative_spatial_cost=use_relative_spatial_cost,
                )

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            bs = B
            tot_loss += loss.item() * bs
            n_train += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(tot_loss/max(1,n_train)):.4f}")
            
        train_loss = tot_loss / max(1, n_train)

        # --------- Validation ---------
        # --------- Validation ---------
        model.eval()
        val_loss, n_val = 0.0, 0
        dice_sum_epoch = 0.0
        f1_sum_epoch = 0.0
        dice_cnt_epoch = 0

        pbar_val = tqdm(dl_val, desc=f"Val   [{len(ds_val)} seqs]", leave=False)
        with torch.no_grad():
            for xb_seq, yb_seq in pbar_val:
                xb_seq = xb_seq.to(device)
                yb_seq = yb_seq.to(device)
                
                B, S, _, T = xb_seq.shape
                N = yb_seq.shape[2]

                logits_seq_list = []
                h = None
                
                # Sequence boyunca tahminleri üret
                for s in range(S):
                    xb_step = xb_seq[:, s, :, :]
                    logits_step, h = model(xb_step, h)
                    logits_seq_list.append(logits_step)
                    
                logits_seq_tensor = torch.stack(logits_seq_list, dim=1) # (B, S, N, T)

                # Sequence PIT Loss hesapla
                loss = sequence_pit_loss_batch(
                    logits_seq_tensor, yb_seq,
                    lambda_bce=lambda_bce,
                    empty_weight=empty_weight,
                    bce_pos_weight=bce_pos_weight,
                    use_dynamic_bce_pos_weight=use_dynamic_bce_pos_weight,
                    use_spatial_cost=use_spatial_cost,
                    spatial_cost_weight=spatial_cost_weight,
                    center_weight=center_weight,
                    duration_weight=duration_weight,
                    start_weight=start_weight,
                    end_weight=end_weight,
                    use_relative_spatial_cost=use_relative_spatial_cost,
                )
                
                bs = B
                val_loss += loss.item() * bs
                n_val += bs

                # Metrikleri hesaplamak için (B, S, N, T) tensorünü (B*S, N, T) şekline düzleştiriyoruz
                logits_flat = logits_seq_tensor.view(B * S, N, T)
                yb_flat = yb_seq.view(B * S, N, T)
                
                # Metrikleri hesapla (her pencere için ayrı ayrı değerlendirilir)
                mean_dice_b, f1_b = evaluate_pit_metrics_batch(logits_flat, yb_flat, thr=0.5, exist_thr=exist_thr)
                
                # bs * S kullanıyoruz çünkü metrik fonksiyonuna B*S adet pencere gönderdik
                dice_sum_epoch += mean_dice_b * (bs * S)
                f1_sum_epoch += f1_b * (bs * S)
                dice_cnt_epoch += (bs * S)

                pbar_val.set_postfix(
                    loss=f"{loss.item():.4f}",
                    dice=f"{mean_dice_b:.3f}",
                    f1=f"{f1_b:.3f}"
                )

        val_loss = val_loss / max(1, n_val)
        val_dice = dice_sum_epoch / max(1, dice_cnt_epoch)
        val_f1   = f1_sum_epoch / max(1, dice_cnt_epoch)

        # Loss history'ye ekle
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        val_f1s.append(val_f1)
        
        print(f"[Epoch {ep:03d}] "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_dice={val_dice:.3f}  "
              f"val_f1={val_f1:.3f}  "
              f"(windows: train={len(ds_train)}, val={len(ds_val)})")
        
        # Loss değişim kontrolü
        if ep > 5:
            recent_train_change = abs(train_losses[-1] - train_losses[-5]) / train_losses[-5]
            recent_val_change = abs(val_losses[-1] - val_losses[-5]) / val_losses[-5]
            if recent_train_change < 0.01 and recent_val_change < 0.01:
                print(f"[Warning] Loss plateaued! Train change: {recent_train_change:.4f}, Val change: {recent_val_change:.4f}")

        # En iyi modeli kaydet (val_loss'a göre)
        if val_loss < best_val:
            best_val = val_loss
            best_report = (val_dice, val_f1)
            best_path = os.path.join(out_dir, f"{model_name}_best.pth")
            torch.save(
                {"state_dict": model.state_dict(), "config": cfg, "val_loss": best_val, "epoch": ep, 
                 "train_losses": train_losses, "val_losses": val_losses, "val_dices": val_dices, "val_f1s": val_f1s},
                best_path,
            )
            print(f"    -> New best model saved! (val_loss: {best_val:.4f})")

        scheduler.step()

    # Son modeli kaydet
    last_path = os.path.join(out_dir, f"{model_name}_last.pth")
    torch.save(
        {"state_dict": model.state_dict(), "config": cfg, "val_loss": val_loss, "epoch": epochs,
         "train_losses": train_losses, "val_losses": val_losses, "val_dices": val_dices, "val_f1s": val_f1s},
        last_path,
    )
    if best_report:
        print(f"Best (by val_loss): Dice={best_report[0]:.3f}, F1={best_report[1]:.3f}")
    print(f"Saved: best -> {os.path.join(out_dir, f'{model_name}_best.pth')}, last -> {last_path}")
    
    # Final training summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    print(f"Best val loss: {best_val:.4f}")
    if len(train_losses) > 10:
        train_improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
        val_improvement = (val_losses[0] - val_losses[-1]) / val_losses[0] * 100
        print(f"Training loss improvement: {train_improvement:.1f}%")
        print(f"Validation loss improvement: {val_improvement:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to JSON config")
    args = parser.parse_args()
    run_training(args.config)