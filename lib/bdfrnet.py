import torch
import torch.nn as nn
import torch.nn.functional as F

from .bdfr_decoders import BDFRDecoder
from .pvtv2 import pvt_v2_b1, pvt_v2_b2


class BDFRNet(nn.Module):
    def __init__(self, num_classes=3, encoder="pvt_v2_b2", pretrained_dir="./pretrained_pth/pvt/", pretrain=True):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 3, 1), nn.BatchNorm2d(3), nn.ReLU(inplace=True))
        if encoder == "pvt_v2_b1":
            self.backbone = pvt_v2_b1()
            weight_path = f"{pretrained_dir}/pvt_v2_b1.pth"
            channels = [512, 320, 128, 64]
        elif encoder == "pvt_v2_b2":
            self.backbone = pvt_v2_b2()
            weight_path = f"{pretrained_dir}/pvt_v2_b2.pth"
            channels = [512, 320, 128, 64]
        else:
            raise ValueError(f"Unsupported BDFRNet encoder: {encoder}")

        if pretrain:
            save_model = torch.load(weight_path, map_location="cpu")
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        self.decoder = BDFRDecoder(channels=channels)
        self.out_head4 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[3], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)

    def forward(self, x, mode="test"):
        if x.size(1) == 1:
            x = self.conv(x)
        x1, x2, x3, x4 = self.backbone(x)
        d4, d3, d2, d1 = self.decoder(x4, [x3, x2, x1])
        p4 = F.interpolate(self.out_head4(d4), scale_factor=16, mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.out_head3(d3), scale_factor=8, mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.out_head2(d2), scale_factor=4, mode="bilinear", align_corners=False)
        p1 = F.interpolate(self.out_head1(d1), scale_factor=4, mode="bilinear", align_corners=False)
        return [p4, p3, p2, p1]
