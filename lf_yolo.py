import torch
import torch.nn as nn
from ultralytics.nn.modules import Detect
#importing simplified backbone
from lf_yolo_backbone import LFYOLO_Simplified

class LFYOLO_WeldDefect(nn.Module):
    def __init__(self, nc=7):
        super().__init__()
        self.yaml = 'lf_yolo_custom.yaml'
        self.backbone = LFYOLO_Simplified()
        # The backbone produces a very large number of channels (6144).
        # Reduce to a practical number (e.g. 256) before the Detect head to
        # avoid huge parameter counts and memory blowups while keeping the
        # backbone architecture unchanged. This 1x1 conv + BN + SiLU mirrors
        # a tiny channel-reduction head.
        self.reduce = nn.Sequential(
            nn.Conv2d(6144, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )

        self.detect = Detect(ch=[256], nc=nc)  # use reduced channel size for head

    def forward(self, x):
        x = self.backbone(x)
        x = self.reduce(x)
        return self.detect([x])

if __name__ == "__main__":
    model = LFYOLO_WeldDefect(nc=7)
    y = model(torch.randn(1, 3, 640, 640))
    print("Output shape:", [o.shape for o in y])
