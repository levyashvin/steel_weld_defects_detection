# LF-YOLO Simplified Backbone
# Author: Yuvay
# Verified clean architecture — Oct 2025
# Default configuration: C=32

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Utility blocks
# ----------------------------
def autopad(k, d=1):
    return ((k - 1) // 2) * d

class CBL(nn.Module):
    """Conv + BN + SiLU"""
    def __init__(self, c1, c2, k=3, s=1, d=1, g=1, act=True):
        super().__init__()
        p = autopad(k, d)
        self.conv = nn.Conv2d(c1, c2, k, s, p, dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class GhostConv(nn.Module):
    """Ghost convolution block"""
    def __init__(self, c1, c2, ratio=2):
        super().__init__()
        hidden = int(c2 / ratio)
        self.primary = CBL(c1, hidden, k=1)
        self.cheap = CBL(hidden, hidden, k=3, g=hidden)
        self.out_ch = c2
        
    def forward(self, x):
        y = self.primary(x)
        g = self.cheap(y)
        return torch.cat([y, g], dim=1)[:, :self.out_ch, :, :]

class GDCConv(nn.Module):
    """Ghost dilated conv"""
    def __init__(self, c, d=1):
        super().__init__()
        self.conv = CBL(c, c, k=3, d=d)
    def forward(self, x):
        return self.conv(x)

# ----------------------------
# Core modules
# ----------------------------
class EFE(nn.Module):
    """Efficient Feature Expansion"""
    def __init__(self, c_in, c_out, ra=0.5):
        super().__init__()
        mid = 2 * c_in
        self.pre = CBL(c_in, mid, k=1)
        self.ra = ra
        self.upper = int(mid * ra)
        self.lower = mid - self.upper
        self.upper_id = nn.Identity()
        self.lower_ghost = GhostConv(self.lower, self.lower)
        self.post = CBL(mid, 2 * c_in, k=1)
        self.reduce = CBL(2 * c_in, c_out, k=1)
        self.shortcut = CBL(c_in, c_out, k=1, act=False) if c_in != c_out else nn.Identity()
    def forward(self, x):
        y = self.pre(x)
        y1, y2 = torch.split(y, [self.upper, self.lower], dim=1)
        y2 = self.lower_ghost(y2)
        y = torch.cat([y1, y2], dim=1)
        y = self.post(y)
        y = self.reduce(y)
        return y + (x if isinstance(self.shortcut, nn.Identity) else self.shortcut(x))

class RMF(nn.Module):
    """Residual Multiscale Fusion"""
    def __init__(self, c_in, c_out):
        super().__init__()
        def pool(k): return nn.MaxPool2d(k, 1, k // 2)
        self.p5, self.p9, self.p13 = pool(5), pool(9), pool(13)

        # Each branch: 3×GDCConv with dilation 1,3,5 — processed in parallel
        self.b5_d1, self.b5_d3, self.b5_d5 = GDCConv(c_in, d=1), GDCConv(c_in, d=3), GDCConv(c_in, d=5)
        self.b9_d1, self.b9_d3, self.b9_d5 = GDCConv(c_in, d=1), GDCConv(c_in, d=3), GDCConv(c_in, d=5)
        self.b13_d1, self.b13_d3, self.b13_d5 = GDCConv(c_in, d=1), GDCConv(c_in, d=3), GDCConv(c_in, d=5)

        self.fuse = CBL(c_in * 10, c_out, k=1)

    def forward(self, x):
        p5, p9, p13 = self.p5(x), self.p9(x), self.p13(x)

        b5 = torch.cat([self.b5_d1(p5), self.b5_d3(p5), self.b5_d5(p5)], dim=1)
        b9 = torch.cat([self.b9_d1(p9), self.b9_d3(p9), self.b9_d5(p9)], dim=1)
        b13 = torch.cat([self.b13_d1(p13), self.b13_d3(p13), self.b13_d5(p13)], dim=1)

        return self.fuse(torch.cat([x, b5, b9, b13], dim=1))
        
# ----------------------------
# LF-YOLO simplified backbone
# ----------------------------
class LFYOLO_Simplified(nn.Module):
    def __init__(self, in_ch=3, C=32):
        super().__init__()
        self.s1 = CBL(in_ch, C)
        self.s2 = nn.MaxPool2d(2)
        self.s3 = EFE(C, 2 * C)  # 32 → 64
        self.s4 = nn.MaxPool2d(2)
        self.s5 = EFE(64, 128)
        self.s6 = EFE(128, 128)
        self.s7 = nn.MaxPool2d(2)
        self.s8 = EFE(128, 256)
        self.s9 = EFE(256, 256)
        self.s10 = RMF(256, 6144)  # matches S20 = 192C = 6144 when C=32

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.s6(x)
        x = self.s7(x)
        x = self.s8(x)
        x = self.s9(x)
        x = self.s10(x)
        return x # Output feature map: [B, 6144, H/8, W/8]

# ----------------------------
# Test
# ----------------------------
if __name__ == "__main__":
    model = LFYOLO_Simplified(in_ch=3, C=32)
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    print("Output shape:", y.shape)