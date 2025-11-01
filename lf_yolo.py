import torch
import torch.nn as nn
from ultralytics.nn.modules import Detect
#importing simplified backbone
from lf_yolo_backbone import LFYOLO_Simplified

class LFYOLO_WeldDefect(nn.Module):
    def __init__(self, nc=7):
        super().__init__()
        self.nc = nc
        self.names = ['air hole', 'bite edge', 'broken arc', 'crack', 'overlap', 'slag inclusion', 'unfused']
        self.stride = torch.tensor([8.0])
        # Provide a minimal Ultralytics-style YAML dict so trainer logic that
        # inspects keys like 'backbone' does not fail even though we inject a
        # prebuilt model. The 'Silence' backbone is a no-op placeholder.
        self.yaml = {
            'nc': nc,
            'names': self.names,
            'backbone': [[-1, 1, 'Silence', []]],
            'head': [[[ -1 ], 1, 'Detect', [nc]]],
        }

        self.backbone = LFYOLO_Simplified()
        self.reduce = nn.Sequential(
            nn.Conv2d(6144, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )

        self.detect = Detect(ch=[256], nc=nc)
        self.detect.stride = self.stride
        self.model = nn.ModuleList([self.backbone, self.reduce, self.detect])

    def forward(self, x):
        x = self.backbone(x)
        x = self.reduce(x)
        return self.detect([x])

if __name__ == "__main__":
    model = LFYOLO_WeldDefect(nc=7)
    y = model(torch.randn(1, 3, 640, 640))
    print("Output shape:", [o.shape for o in y])
