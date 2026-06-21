import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhanceCAB(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1, bias=False))
        self.max_mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weight = self.sigmoid(self.avg_mlp(self.avg_pool(x)) + self.max_mlp(self.max_pool(x)))
        return x * weight + x


class CAB(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.bn(self.conv(torch.cat([avg, maxv], dim=1))))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel = CAB(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.channel(x))


class WaveletHighFreq(nn.Module):
    def __init__(self, wavelet="db1"):
        super().__init__()
        if wavelet != "db1":
            raise ValueError("Only db1/Haar wavelet is implemented with the fast torch path.")

    def forward(self, x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        lh = (x00 + x01 - x10 - x11) * 0.5
        hl = (x00 - x01 + x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return torch.cat([lh, hl, hh], dim=1)


class WaveletLowFreq(nn.Module):
    def __init__(self, wavelet="db1"):
        super().__init__()
        if wavelet != "db1":
            raise ValueError("Only db1/Haar wavelet is implemented with the fast torch path.")

    def forward(self, x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        return (x00 + x01 + x10 + x11) * 0.5


class HighFreqBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.wavelet = WaveletHighFreq()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.up = nn.ConvTranspose2d(channels, channels, 2, stride=2)
        self.cab = CAB(channels)

    def forward(self, x):
        return self.cab(self.up(self.edge_conv(self.wavelet(x))))


class LowFreqBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.wavelet = WaveletLowFreq()
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 5, padding=2, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.up = nn.ConvTranspose2d(channels, channels, 2, stride=2)
        self.cab = CAB(channels)

    def forward(self, x):
        return self.cab(self.up(self.conv(self.wavelet(x))))


class ContextBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.cab = CAB(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.cab(self.bn(self.conv(x))))
        return x * attention


class DDBE(nn.Module):
    def __init__(self, in_channels, out_channels, use_upsample=True):
        super().__init__()
        self.use_upsample = use_upsample
        self.ecab = EnhanceCAB(in_channels)
        self.high = HighFreqBranch(in_channels)
        self.low = LowFreqBranch(in_channels)
        self.context = ContextBranch(in_channels)
        self.attention = CBAM(in_channels * 3, reduction=4)
        if use_upsample:
            self.out = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels * 3, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        else:
            self.out = nn.Sequential(nn.Conv2d(in_channels * 3, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.ecab(x)
        fused = self.attention(torch.cat([self.high(x), self.low(x), self.context(x)], dim=1))
        return self.out(fused)


class ACAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.conv_mask_g = nn.Conv2d(channels, 1, 1)
        self.conv_mask_x = nn.Conv2d(channels, 1, 1)
        self.softmax = nn.Softmax(dim=2)
        self.g_transform = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.LayerNorm([hidden, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1))
        self.x_transform = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.LayerNorm([hidden, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1))
        self.sigmoid = nn.Sigmoid()

    def _context(self, x, conv_mask):
        b, c, _, _ = x.shape
        mask = self.softmax(conv_mask(x).view(b, 1, -1))
        context = torch.matmul(x.view(b, c, -1), mask.permute(0, 2, 1))
        return context.view(b, c, 1, 1)

    def forward(self, g, x):
        wg = self.sigmoid(self.g_transform(self._context(g, self.conv_mask_g)))
        wx = self.sigmoid(self.x_transform(self._context(x, self.conv_mask_x)))
        return g * wg + x * wx


class BDFRDecoder(nn.Module):
    def __init__(self, channels=(512, 320, 128, 64)):
        super().__init__()
        self.ddbe4 = DDBE(channels[0], channels[1], use_upsample=True)
        self.acam3 = ACAM(channels[1])
        self.ddbe3 = DDBE(channels[1], channels[2], use_upsample=True)
        self.acam2 = ACAM(channels[2])
        self.ddbe2 = DDBE(channels[2], channels[3], use_upsample=True)
        self.acam1 = ACAM(channels[3])
        self.ddbe1 = DDBE(channels[3], channels[3], use_upsample=False)

    def forward(self, x4, skips):
        x3, x2, x1 = skips
        d4 = self.ddbe4(x4)
        d3 = self.ddbe3(self.acam3(d4, x3))
        d2 = self.ddbe2(self.acam2(d3, x2))
        d1 = self.ddbe1(self.acam1(d2, x1))
        return [d4, d3, d2, d1]
