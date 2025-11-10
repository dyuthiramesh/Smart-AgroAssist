# model.py
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

# === Attention modules ===
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))


class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# === Dual-Domain Model (RGB + FFT) ===
class DualDomainModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()

        # Frequency domain CNN
        self.freq_conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Attention fusion
        self.cbam = CBAM(in_planes=in_features + 32)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features + 32, num_classes)
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, rgb, freq):
        feat_rgb = self.backbone(rgb)                 # [B, in_features]
        feat_freq = self.freq_conv(freq).view(freq.size(0), -1)  # [B, 32]
        fused = torch.cat([feat_rgb, feat_freq], dim=1)         # [B, in_features+32]
        fused = fused.unsqueeze(-1).unsqueeze(-1)               # [B, C, 1, 1]
        attn = self.cbam(fused).view(fused.size(0), -1)         # [B, C]
        return self.classifier(attn)                           # [B, num_classes]
