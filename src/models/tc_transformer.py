import torch, torch.nn as nn
from einops import rearrange

class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c, max(4, c//r)), nn.ReLU(inplace=True),
            nn.Linear(max(4, c//r), c), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w.unsqueeze(-1)

class ResConv1DBlock(nn.Module):
    def __init__(self, c, k=7, p=None):
        super().__init__()
        pad = (k//2) if p is None else p
        self.net = nn.Sequential(
            nn.Conv1d(c, c, k, padding=pad),
            nn.BatchNorm1d(c), nn.GELU(),
            nn.Conv1d(c, c, k, padding=pad),
            nn.BatchNorm1d(c)
        )
        self.act = nn.GELU()
        self.se = SEBlock(c)
    def forward(self, x):
        y = self.net(x)
        y = self.se(y)
        return self.act(x+y)

class ConvStem(nn.Module):
    def __init__(self, chs=(32,64,128), k=7, in_channels=1):
        super().__init__()
        layers=[]
        in_c=in_channels
        for c in chs:
            layers += [
                nn.Conv1d(in_c, c, k, padding=k//2),
                nn.BatchNorm1d(c), nn.GELU(),
                ResConv1DBlock(c, k=k),
                nn.MaxPool1d(2)
            ]
            in_c=c
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        import math, torch
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, L, D]
    def forward(self, x):
        # x: [B, L, D]
        L = x.size(1)
        return x + self.pe[:,:L,:]

class TCTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        chs = cfg["cnn_channels"]; k = cfg["kernel_size"]
        in_ch = cfg.get("in_channels", 1)   # NEW
        self.stem = ConvStem(tuple(chs), k=k, in_channels=in_ch)
        d_model = cfg["transformer"]["d_model"]
        self.proj = nn.Conv1d(chs[-1], d_model, 1)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=cfg["transformer"]["n_heads"],
            dim_feedforward=d_model*cfg["transformer"]["ff_mult"],
            dropout=cfg.get("dropout",0.1), batch_first=True, activation="gelu"
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=cfg["transformer"]["num_layers"])
        out_dim = 4 if cfg.get("loss","smoothl1")=="hetero" else 2  # mean/logvar or plain
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(d_model, cfg["head_hidden"]), nn.GELU(),
            nn.Dropout(cfg.get("dropout",0.1)),
            nn.Linear(cfg["head_hidden"], out_dim)
        )

    def forward(self, x):
        # x: [B,C,L]
        z = self.stem(x)                # [B,C',L']
        z = self.proj(z)                # [B,D,L']
        zt = rearrange(z, "b d l -> b l d")
        zt = self.pos(zt)
        zt = self.tr(zt)                # [B,L',D]
        zt = rearrange(zt, "b l d -> b d l")
        out = self.head(zt)             # [B,2] or [B,4]
        return out


    
