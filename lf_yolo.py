import torch
import torch.nn as nn
from ultralytics.nn.modules import Detect
#importing simplified backbone
from lf_yolo_backbone import LFYOLO_Simplified

class LFYOLO_WeldDefect(nn.Module):
    def __init__(self, nc=7):
        super().__init__()
        self.backbone = LFYOLO_Simplified()
        self.detect = Detect(ch=[6144], nc=nc)  # 6144 from your RMF output channels

    def forward(self, x):
        x = self.backbone(x)
        return self.detect([x])

if __name__ == "__main__":
    model = LFYOLO_WeldDefect(nc=7)
    y = model(torch.randn(1, 3, 640, 640))
    print("Output shape:", [o.shape for o in y])
