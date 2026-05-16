# pri_train.py
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from pri_net import PRINet

# -------- Dataset --------
class PRIDataset(Dataset):
    def __init__(self, X, y_cls, y_logpri=None, mean=None, std=None):
        self.X = X.astype(np.float32)
        self.y_cls = y_cls.astype(np.int64)
        self.y_logpri = None if y_logpri is None else y_logpri.astype(np.float32)
        # z-score
        if mean is not None and std is not None:
            self.mean = mean.astype(np.float32)
            self.std = np.where(std == 0, 1.0, std).astype(np.float32)
            self.X = (self.X - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx):
        xi = self.X[idx]
        yi = self.y_cls[idx]
        if self.y_logpri is not None:
            yr = self.y_logpri[idx]
        else:
            yr = None
        sample = {"x": torch.from_numpy(xi), "y_cls": torch.tensor(yi)}
        if yr is not None:
            sample["y_logpri"] = torch.tensor(yr, dtype=torch.float32)
        else:
            sample["y_logpri"] = None
        return sample

# -------- Metrics --------
def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def mae_mape(exp_pred, exp_true, eps=1e-6):
    # exp_: µs cinsine geri çevrilmiş değerler
    abs_err = np.abs(exp_pred - exp_true)
    mae = float(abs_err.mean())
    mape = float((abs_err / np.maximum(np.abs(exp_true), eps)).mean() * 100.0)
    return mae, mape

# -------- Training --------
def train_loop(
    X, y, classes_json,
    y_logpri=None,
    out_dir="out/pri_model",
    input_dim=1106,
    hidden="512,256,128",
    dropout=0.1,
    lr=3e-4,
    weight_decay=0.0,
    batch_size=64,
    epochs=50,
    lambda_reg=0.5,
    huber_delta=1.0,
    val_size=0.2,
    seed=42,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Stratified split (sınıf dengesini koru)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    (train_idx, val_idx) = next(splitter.split(X, y))

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    if y_logpri is not None:
        yreg_tr, yreg_val = y_logpri[train_idx], y_logpri[val_idx]
    else:
        yreg_tr = yreg_val = None

    # Z-score scaler (train set)
    mean = X_tr.mean(axis=0, keepdims=True)
    std = X_tr.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)

    # Dataset & Loader
    ds_tr = PRIDataset(X_tr, y_tr, yreg_tr, mean, std)
    ds_val = PRIDataset(X_val, y_val, yreg_val, mean, std)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)

    # Sınıf ağırlıkları (imbalans için)
    num_classes = len(json.load(open(classes_json, "r", encoding="utf-8"))["classes"])
    class_counts = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
    inv = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = inv / inv.sum() * num_classes
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # Model
    hidden_dims = tuple(int(v) for v in hidden.split(",") if v.strip())
    do_reg = y_logpri is not None
    model = PRINet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        do_regression=do_reg
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_path = out_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, tot_acc, n_batches = 0.0, 0.0, 0
        for batch in dl_tr:
            x = batch["x"].to(device)
            y_cls = batch["y_cls"].to(device)
            if do_reg:
                y_logpri_b = batch["y_logpri"].to(device)
            else:
                y_logpri_b = None

            targets = {"y_cls": y_cls}
            if do_reg: targets["y_logpri"] = y_logpri_b

            out = model(
                x, targets=targets, class_weights=class_weights_t,
                lambda_reg=lambda_reg, huber_delta=huber_delta
            )
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            tot_loss += loss.item()
            tot_acc += accuracy(out["logits"], y_cls)
            n_batches += 1

        tr_loss = tot_loss / max(1, n_batches)
        tr_acc = tot_acc / max(1, n_batches)

        # Validation
        model.eval()
        val_loss, val_acc, n_val = 0.0, 0.0, 0
        preds_reg, gts_reg = [], []
        with torch.no_grad():
            for batch in dl_val:
                x = batch["x"].to(device)
                y_cls = batch["y_cls"].to(device)
                if do_reg:
                    y_logpri_b = batch["y_logpri"].to(device)
                else:
                    y_logpri_b = None

                targets = {"y_cls": y_cls}
                if do_reg: targets["y_logpri"] = y_logpri_b

                out = model(
                    x, targets=targets, class_weights=class_weights_t,
                    lambda_reg=lambda_reg, huber_delta=huber_delta
                )
                loss = out["loss"]
                val_loss += loss.item()
                val_acc += accuracy(out["logits"], y_cls)
                n_val += 1

                if do_reg:
                    preds_reg.append(out["logpri_pred"].cpu().numpy())
                    gts_reg.append(y_logpri_b.cpu().numpy())

        val_loss = val_loss / max(1, n_val)
        val_acc = val_acc / max(1, n_val)

        msg = f"[Epoch {epoch:03d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f}"

        if do_reg and preds_reg:
            pr = np.concatenate(preds_reg, axis=0).squeeze()
            gt = np.concatenate(gts_reg, axis=0).squeeze()
            # µs'e geri çevir
            pr_us = np.exp(pr)
            gt_us = np.exp(gt)
            mae, mape = mae_mape(pr_us, gt_us)
            msg += f" | val_PRI_MAE={mae:.2f}us val_PRI_MAPE={mape:.2f}%"

        print(msg)

        # early stopping: en iyi val_loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "hidden": hidden,
                "dropout": dropout,
                "num_classes": num_classes,
                "do_regression": do_reg,
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
                "classes": json.load(open(classes_json, "r", encoding="utf-8")),
                # Validation set for inference
                "X_val": X_val,
                "y_val": y_val,
                "y_logpri_val": yreg_val
            }, best_path)

    print(f"[OK] Best model saved to {best_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Folder with X_*.npy, y_labels.npy, classes.json")
    ap.add_argument("--feature_file", type=str, default="X_feat_vector.npy")
    ap.add_argument("--y_file", type=str, default="y_labels.npy")
    ap.add_argument("--y_logpri_file", type=str, default="y_logpri.npy")
    ap.add_argument("--out_dir", type=str, default="out/pri_model")
    ap.add_argument("--input_dim", type=int, default=1106)
    ap.add_argument("--hidden", type=str, default="512,256,128")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lambda_reg", type=float, default=0.5)
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    X = np.load(data_dir / args.feature_file)
    y = np.load(data_dir / args.y_file)
    classes_json = str(data_dir / "classes.json")
    y_logpri_path = data_dir / args.y_logpri_file
    y_logpri = np.load(y_logpri_path) if y_logpri_path.exists() else None

    # --> OTOMATİK BOYUT ALGILAMA <--
    input_dim = X.shape[1] if args.input_dim <= 0 else args.input_dim
    print(f"[Info] Feature dim detected: {input_dim} (X.shape={X.shape})")

    train_loop(
        X, y, classes_json,
        y_logpri=y_logpri,
        out_dir=args.out_dir,
        input_dim=args.input_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lambda_reg=args.lambda_reg,
        huber_delta=args.huber_delta,
        val_size=args.val_size,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
