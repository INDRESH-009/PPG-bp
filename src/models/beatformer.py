import torch, torch.nn as nn
from einops import rearrange

class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(c, max(4, c//r)), nn.ReLU(inplace=True),
            nn.Linear(max(4, c//r), c), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w.unsqueeze(-1)

class BeatEncoder(nn.Module):
    """CNN encoder for a single beat: input [B,C,T] -> [B,D]"""
    def __init__(self, in_ch=3, chs=(32,64,128), k=7, se_reduction=8, out_dim=128, dropout=0.1):
        super().__init__()
        layers=[]; c_in=in_ch
        for c in chs:
            layers += [
                nn.Conv1d(c_in, c, k, padding=k//2),
                nn.BatchNorm1d(c), nn.GELU(),
                nn.Conv1d(c, c, k, padding=k//2),
                nn.BatchNorm1d(c), nn.GELU(),
                SEBlock(c, r=se_reduction),
                nn.MaxPool1d(2)
            ]
            c_in=c
        self.cnn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(chs[-1], out_dim), nn.GELU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        z = self.cnn(x)         # [B,C',T']
        z = self.head(z)        # [B,D]
        return z

class BeatFormer(nn.Module):
    """
    Input:
      beats: [B, MaxBeats, C, T]
      mask:  [B, MaxBeats]  (1 valid, 0 pad)
    Output:
      y:     [B,2] or [B,4] (hetero)
    """
    def __init__(self, cfg):
        super().__init__()
        in_ch = cfg.get("in_channels", 3)
        chs = cfg["cnn_channels"]; k = cfg["kernel_size"]
        d_model = cfg["transformer"]["d_model"]
        self.encoder = BeatEncoder(in_ch, tuple(chs), k, cfg["se_reduction"],
                                   out_dim=d_model, dropout=cfg.get("dropout",0.1))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=cfg["transformer"]["n_heads"],
            dim_feedforward=d_model*cfg["transformer"]["ff_mult"],
            dropout=cfg.get("dropout",0.1), batch_first=True, activation="gelu"
        )
        self.trans = nn.TransformerEncoder(enc_layer, num_layers=cfg["transformer"]["num_layers"])
        self.pool = nn.AdaptiveAvgPool1d(1)  # we'll pool across sequence with mask-aware mean
        out_dim = 4 if cfg.get("loss","smoothl1")=="hetero" else 2
        self.head = nn.Sequential(
            nn.Linear(d_model, cfg["head_hidden"]), nn.GELU(),
            nn.Dropout(cfg.get("dropout",0.1)),
            nn.Linear(cfg["head_hidden"], out_dim)
        )

    def forward(self, beats, mask):
        # beats: [B, M, C, T], mask: [B, M]
        B, M, C, T = beats.shape
        beats = beats.view(B*M, C, T)
        emb = self.encoder(beats)           # [B*M, D]
        D = emb.size(-1)
        emb = emb.view(B, M, D)             # [B, M, D]

        # apply padding mask: True means to MASK (ignore)
        src_key_padding_mask = (mask < 0.5)  # [B, M] booleans
        z = self.trans(emb, src_key_padding_mask=src_key_padding_mask)  # [B, M, D]

        # masked mean over sequence
        m = mask.unsqueeze(-1)              # [B, M, 1]
        sum_z = torch.sum(z*m, dim=1)       # [B, D]
        denom = torch.clamp(m.sum(dim=1), min=1e-6)
        z_pool = sum_z / denom              # [B, D]
        out = self.head(z_pool)             # [B, 2 or 4]
        return out
