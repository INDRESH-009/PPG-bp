import torch
import torch.nn as nn

# -------- Building blocks --------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, s=1, p=None, act="gelu"):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU() if act == "gelu" else nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        mid = max(4, c // r)
        self.fc = nn.Sequential(nn.Linear(c, mid), nn.ReLU(inplace=True), nn.Linear(mid, c), nn.Sigmoid())
    def forward(self, x):
        w = self.pool(x).squeeze(-1)           # [B, C]
        w = self.fc(w).unsqueeze(-1)           # [B, C, 1]
        return x * w

class ResBlock(nn.Module):
    def __init__(self, c, k=7, se_reduction=8, act="gelu"):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(c, c, k=k, act=act),
            nn.Conv1d(c, c, kernel_size=k, padding=k//2),
            nn.BatchNorm1d(c)
        )
        self.se = SEBlock(c, r=se_reduction)
        self.act = nn.GELU() if act == "gelu" else nn.ReLU(inplace=True)
    def forward(self, x):
        y = self.net(x)
        y = self.se(y)
        return self.act(x + y)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, se_reduction=8, act="gelu"):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.in_proj = ConvBNAct(in_ch, out_ch, k=3, act=act)
        self.res = ResBlock(out_ch, k=k, se_reduction=se_reduction, act=act)
    def forward(self, x):
        x = self.pool(x)
        x = self.in_proj(x)
        x = self.res(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, k=7, se_reduction=8, act="gelu"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.fuse = ConvBNAct(in_ch + skip_ch, out_ch, k=3, act=act)
        self.res = ResBlock(out_ch, k=k, se_reduction=se_reduction, act=act)
    def forward(self, x, skip):
        x = self.up(x)
        # match length (can differ by 1 due to pooling)
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            pad_left = diff // 2
            pad_right = diff - pad_left
            x = nn.functional.pad(x, (pad_left, pad_right))
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.res(x)
        return x

# -------- UNet1D with SE and residuals --------
class UNet1D(nn.Module):
    """
    A 1D U-Net for PPG regression.
    Inputs:
      x: [B, C_in, L]
    Outputs:
      y: [B, 2] or [B, 4]  (if loss == 'hetero': [mu_sbp, mu_dbp, logvar_sbp, logvar_dbp])
    """
    def __init__(self, cfg):
        super().__init__()
        C_in = cfg.get("in_channels", 1)
        chans = cfg.get("cnn_channels", [32, 64, 128, 256])  # deeper than TC for better multi-scale
        k = cfg.get("kernel_size", 7)
        se = cfg.get("se_reduction", 8)
        act = "gelu"
        self.loss_kind = cfg.get("loss", "smoothl1")

        # Encoder
        self.enc0 = nn.Sequential(ConvBNAct(C_in, chans[0], k=3, act=act), ResBlock(chans[0], k=k, se_reduction=se, act=act))
        self.enc1 = Down(chans[0], chans[1], k=k, se_reduction=se, act=act)
        self.enc2 = Down(chans[1], chans[2], k=k, se_reduction=se, act=act)
        self.enc3 = Down(chans[2], chans[3], k=k, se_reduction=se, act=act)

        # Bottleneck (dilated to widen RF)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(chans[3], chans[3], kernel_size=k, padding=(k//2)*1, dilation=1),
            nn.BatchNorm1d(chans[3]), nn.GELU(),
            nn.Conv1d(chans[3], chans[3], kernel_size=k, padding=(k//2)*2, dilation=2),
            nn.BatchNorm1d(chans[3]), nn.GELU(),
            SEBlock(chans[3], r=se)
        )

        # Decoder
        self.up2 = Up(chans[3], chans[2], chans[2], k=k, se_reduction=se, act=act)
        self.up1 = Up(chans[2], chans[1], chans[1], k=k, se_reduction=se, act=act)
        self.up0 = Up(chans[1], chans[0], chans[0], k=k, se_reduction=se, act=act)

        # Head: global pooling over time + MLP
        out_dim = 4 if self.loss_kind == "hetero" else 2
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(chans[0], cfg.get("head_hidden", 256)), nn.GELU(),
            nn.Dropout(cfg.get("dropout", 0.2)),
            nn.Linear(cfg.get("head_hidden", 256), out_dim)
        )

    def forward(self, x):
        # x: [B, C_in, L]
        s0 = self.enc0(x)        # [B, c0, L]
        s1 = self.enc1(s0)       # [B, c1, L/2]
        s2 = self.enc2(s1)       # [B, c2, L/4]
        s3 = self.enc3(s2)       # [B, c3, L/8]
        b  = self.bottleneck(s3) # [B, c3, L/8]
        d2 = self.up2(b, s2)     # [B, c2, L/4]
        d1 = self.up1(d2, s1)    # [B, c1, L/2]
        d0 = self.up0(d1, s0)    # [B, c0, L]
        y  = self.head(d0)       # [B, 2 or 4]
        return y
