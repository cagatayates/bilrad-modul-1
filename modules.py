import torch
import torch.nn as nn
import torch.nn.functional as F

def _norm_layer(norm: str, num_channels: int, groups_gn: int = 8):
    if norm.lower() == "bn":
        return nn.BatchNorm1d(num_channels)
    elif norm.lower() == "gn":
        g = min(groups_gn, num_channels)
        while num_channels % g != 0 and g > 1:
            g -= 1
        if g <= 1:
            return nn.GroupNorm(1, num_channels)
        return nn.GroupNorm(g, num_channels)
    else:
        return nn.Identity()

class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B, C, L)
        y = self.global_pool(x)  # (B, C, 1)
        y = self.fc(y)           # (B, C, 1)
        return x * y


class StatefulChannelAttention1D(nn.Module):
    """
    Pencereler arası (inter-window) hafızayı tutan, GRU tabanlı kanal dikkat mekanizması.
    Agile emiterlerin takibi ve kanal kaymasını (track swap) önlemek için tasarlandı.
    """
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        reduced_ch = max(1, in_channels // reduction)
        
        self.fc1 = nn.Conv1d(in_channels, reduced_ch, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        # Pencereler arası hafıza (Memory) için GRU hücresi
        self.gru_cell = nn.GRUCell(input_size=reduced_ch, hidden_size=reduced_ch)
        
        self.fc2 = nn.Conv1d(reduced_ch, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, h=None):
        # x: (B, C, L)
        B, C, L = x.shape
        
        # Global bağlamı (context) çıkart
        y = self.global_pool(x)               # (B, C, 1)
        y = self.fc1(y)                       # (B, C/r, 1)
        y = self.relu(y).squeeze(-1)          # (B, C/r)
        
        # Recurrent (Hafıza) Adımı
        if h is None:
            # Eğer ilk pencereyse (hidden state yoksa), sıfırlarla başlat
            h = torch.zeros_like(y, device=y.device)
            
        # Hafızayı güncelle
        h_new = self.gru_cell(y, h)           # (B, C/r)
        
        # Yeni hafızayı kullanarak kanalları ağırlıklandır (Attention)
        y_out = h_new.unsqueeze(-1)           # (B, C/r, 1)
        y_out = self.fc2(y_out)               # (B, C, 1)
        y_out = self.sigmoid(y_out)
        
        # Girdi feature map'ini ağırlıklarla çarp ve yeni hafızayı döndür
        return x * y_out, h_new

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, norm="gn", dropout=0.0, groups_gn: int = 8, residual: bool = False):
        super().__init__()
        pad = ((k - 1) // 2) * d
        self.residual = residual
        
        # First convolution
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=d, bias=False)
        self.n1 = _norm_layer(norm, out_ch, groups_gn)
        
        # Second convolution
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, dilation=d, bias=False)
        self.n2 = _norm_layer(norm, out_ch, groups_gn)
        
        # Projection for residual connection if needed
        if self.residual and in_ch != out_ch:
            self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.proj = None
            
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.n1(out)
        out = F.relu(out, inplace=True)
        out = self.drop(out)
        
        out = self.conv2(out)
        out = self.n2(out)
        
        # Residual connection
        if self.residual:
            if self.proj is not None:
                identity = self.proj(identity)
            out = out + identity
            
        out = F.relu(out, inplace=True)
        return out

class UpBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, norm="gn", dropout=0.0, up_mode="linear", groups_gn: int = 8, residual: bool = False):
        super().__init__()
        if up_mode == "transposed":
            self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        elif up_mode == "linear":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                nn.Conv1d(in_ch, out_ch, kernel_size=1)
            )
        else:
            raise ValueError("up_mode must be 'transposed' or 'linear'")
        self.conv = ConvBlock1D(out_ch * 2, out_ch, k=k, d=1, norm=norm, dropout=dropout, groups_gn=groups_gn, residual=residual)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size(-1) != skip.size(-1):
            L = min(x.size(-1), skip.size(-1))
            x = x[..., :L]
            skip = skip[..., :L]
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet1D(nn.Module):
    """
    Input:  (B, in_channels=2, L)
    Output: logits (B, num_emitters=N, L)
    """
    def __init__(self,
                 in_channels: int = 2,
                 num_emitters: int = 3,
                 base_ch: int = 64,
                 depth: int = 4,
                 norm: str = "gn",
                 dropout: float = 0.1,
                 up_mode: str = "linear",
                 bottleneck_dilation: int = 4,
                 encoder_kernel_sizes = None,
                 groups_gn: int = 8,
                 use_residual: bool = False,
                 attention_type: str = "stateful",
                 return_state: bool = True):
        super().__init__()
        assert depth >= 1
        if encoder_kernel_sizes is None:
            encoder_kernel_sizes = [3] * depth
        assert len(encoder_kernel_sizes) == depth

        # Encoder
        chs = [base_ch * (2 ** i) for i in range(depth)]
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for i, c in enumerate(chs):
            k = encoder_kernel_sizes[i]
            self.enc_blocks.append(ConvBlock1D(prev, c, k=k, d=1, norm=norm, dropout=dropout, groups_gn=groups_gn, residual=use_residual))
            self.pools.append(nn.MaxPool1d(kernel_size=2))
            prev = c

        # Bottleneck
        self.bottleneck = ConvBlock1D(chs[-1], chs[-1] * 2, k=3, d=bottleneck_dilation, norm=norm, dropout=dropout, groups_gn=groups_gn, residual=False)
        self.return_state = bool(return_state)
        att = (attention_type or "stateful").lower()
        if att in ("classic", "ca", "channel", "channel_attention", "channelattention", "simple"):
            self.bottleneck_attention = ChannelAttention1D(chs[-1] * 2, reduction=8)
            self._attention_is_stateful = False
        elif att in ("stateful", "gru", "recurrent", "memory"):
            self.bottleneck_attention = StatefulChannelAttention1D(chs[-1] * 2, reduction=8)
            self._attention_is_stateful = True
        else:
            raise ValueError(f"Unknown attention_type: {attention_type!r}")
        
        # Decoder
        dec_in = [chs[-1] * 2] + [chs[-1] // (2 ** i) for i in range(depth - 1)]
        dec_out = [chs[-1]] + [chs[-1] // (2 ** (i + 1)) for i in range(depth - 1)]
        self.up_blocks = nn.ModuleList()
        dec_k = list(reversed(encoder_kernel_sizes))
        for cin, cout, k in zip(dec_in, dec_out, dec_k):
            self.up_blocks.append(UpBlock1D(cin, cout, k=k, norm=norm, dropout=dropout, up_mode=up_mode, groups_gn=groups_gn, residual=use_residual))

        # Head
        self.head = nn.Conv1d(dec_out[-1], num_emitters, kernel_size=1)

    def forward(self, x, h=None):
        skips = []
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
            
        x = self.bottleneck(x)
        
        if self._attention_is_stateful:
            x, h_new = self.bottleneck_attention(x, h)
        else:
            x = self.bottleneck_attention(x)
            h_new = h
        
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)
            
        logits = self.head(x)  # (B, N, L)
        
        if self.return_state:
            return logits, h_new
        return logits
