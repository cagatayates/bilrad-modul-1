# pri_net.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.1, use_layernorm=True):
        super().__init__()
        self.fc = nn.Linear(in_ch, out_ch)
        self.act = nn.GELU()
        self.do = nn.Dropout(p_drop)
        self.ln = nn.LayerNorm(out_ch) if use_layernorm else nn.Identity()

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.do(x)
        x = self.ln(x)
        return x

class PRINet(nn.Module):
    """
    Girdi: feat_vector (örn. 1156)
    Çıkışlar:
      - mode logits: (B, num_classes)
      - log-PRI: (B, 1)  [opsiyonel loss]
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims=(512, 256, 128),
        dropout=0.1,
        use_layernorm=True,
        do_regression=True
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(MLPBlock(dims[i], dims[i+1], p_drop=dropout, use_layernorm=use_layernorm))
        self.backbone = nn.Sequential(*blocks)
        last_dim = dims[-1]
        self.head_cls = nn.Linear(last_dim, num_classes)
        self.do_regression = do_regression
        if do_regression:
            self.head_reg = nn.Linear(last_dim, 1)  # log-PRI (µs) regresyonu

    def forward(self, x, targets=None, class_weights=None, lambda_reg=0.5, huber_delta=1.0):
        """
        targets: dict olabilir:
          - "y_cls": (B,) int64
          - "y_logpri": (B,) float32  [opsiyonel]
        """
        h = self.backbone(x)
        logits = self.head_cls(h)
        out = {"logits": logits}

        loss = None
        losses = {}

        if targets is not None and "y_cls" in targets:
            y = targets["y_cls"].long()
            if class_weights is not None:
                ce = F.cross_entropy(logits, y, weight=class_weights)
            else:
                ce = F.cross_entropy(logits, y)
            losses["ce"] = ce
            loss = ce if loss is None else loss + ce

        if self.do_regression and targets is not None and ("y_logpri" in targets) and (targets["y_logpri"] is not None):
            yreg = targets["y_logpri"].view(-1, 1)
            pred = self.head_reg(h)
            # Huber (Smooth L1)
            reg = F.huber_loss(pred, yreg, delta=huber_delta)
            losses["reg"] = reg
            if "ce" in losses:
                loss = (1.0 - lambda_reg) * losses["ce"] + lambda_reg * reg
            else:
                loss = reg
            out["logpri_pred"] = pred
        elif self.do_regression:
            out["logpri_pred"] = self.head_reg(h)

        out["loss"] = loss
        out["losses"] = losses
        return out
