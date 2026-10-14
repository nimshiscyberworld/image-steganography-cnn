import os
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_DIR = "/kaggle/working/paper_secret_image_reproduction"
os.makedirs(PROJECT_DIR, exist_ok=True)

model_code = r'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperEdgeDetector(nn.Module):
    """
    VGG-style edge detector with five side outputs and
    a learnable weighted-fusion layer.

    This is an implementation scaffold based on the paper's
    described architecture, not yet a verified exact reproduction.
    """

    def __init__(self):
        super().__init__()

        # Five feature stages; channels follow a VGG-style design.
        channels = [64, 128, 256, 512, 512]
        inputs = [3, 64, 128, 256, 512]
        blocks = [2, 2, 3, 3, 3]

        self.stages = nn.ModuleList()
        for in_ch, out_ch, count in zip(inputs, channels, blocks):
            layers = []
            for i in range(count):
                layers.extend([
                    nn.Conv2d(
                        in_ch if i == 0 else out_ch,
                        out_ch,
                        kernel_size=3,
                        padding=1
                    ),
                    nn.ReLU(inplace=True)
                ])
            self.stages.append(nn.Sequential(*layers))

        self.pools = nn.ModuleList([
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
            for _ in range(4)
        ])

        # One side output per feature stage.
        self.side_heads = nn.ModuleList([
            nn.Conv2d(ch, 1, kernel_size=1)
            for ch in channels
        ])

        # Learnable fusion of the five side outputs.
        self.fusion = nn.Conv2d(5, 1, kernel_size=1, bias=False)

    def forward(self, x):
        original_size = x.shape[-2:]
        side_outputs = []

        for i, stage in enumerate(self.stages):
            x = stage(x)
            side = self.side_heads[i](x)
            side = F.interpolate(
                side,
                size=original_size,
                mode="bilinear",
                align_corners=False
            )
            side_outputs.append(side)

            if i < len(self.pools):
                x = self.pools[i](x)

        fused = self.fusion(torch.cat(side_outputs, dim=1))
        return side_outputs, fused
'''

MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

file_path = os.path.join(MODEL_DIR, "edge_detector.py")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(model_code)

print("Saved:", file_path)