import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

class LiteUNetDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels=1):
        super().__init__()
        # features = [c3, c2, c1] (deep to shallow)
        # c3: features.12 (576)
        # c2: features.3 (24)
        # c1: features.1 (16)
        
        c3_ch, c2_ch, c1_ch = encoder_channels
        
        # Up 1: c3 (1/32) -> c2 size (1/8)
        # 576 + 24 = 600
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False) # 1/32 -> 1/8
        self.conv1 = nn.Sequential(
            nn.Conv2d(c3_ch + c2_ch, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Up 2: 1/8 -> c1 size (1/4)
        # 128 + 16 = 144
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # 1/8 -> 1/4
        self.conv2 = nn.Sequential(
            nn.Conv2d(128 + c1_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Up 3: 1/4 -> 1/1
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, features):
        # features = [c1, c2, c3] (shallow to deep)
        c1, c2, c3 = features
        
        x = self.up1(c3)
        # Interpolate if exact match needed (e.g. if input size odd)
        if x.size()[2:] != c2.size()[2:]:
            x = F.interpolate(x, size=c2.size()[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, c2], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        if x.size()[2:] != c1.size()[2:]:
            x = F.interpolate(x, size=c1.size()[2:], mode='bilinear', align_corners=False)
            
        x = torch.cat([x, c1], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        return self.final_conv(x)

class CardiacGuidanceNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Encoder: MobileNetV3-Small with Feature Extraction
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        
        return_nodes = {
            "features.1": "c1",   # 16 ch, 1/4
            "features.3": "c2",   # 24 ch, 1/8
            "features.12": "c3"   # 576 ch, 1/32
        }
        self.encoder = create_feature_extractor(backbone, return_nodes=return_nodes)
        
        # Channels
        c1_ch = 16 
        c2_ch = 24
        c3_ch = 576
        
        # Segmentation decoder
        self.decoder = LiteUNetDecoder([c3_ch, c2_ch, c1_ch], out_channels=1)
        
        # Aux Head (on c2)
        self.aux_head = nn.Conv2d(c2_ch, 1, kernel_size=1)
        
        # View head (on c3)
        self.view_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3_ch, 128),
            nn.ReLU(),
            nn.Linear(128, 3) 
        )

    def forward(self, x):
        input_size = x.size()[2:]
        
        # Encoder
        features = self.encoder(x)
        c1 = features["c1"]
        c2 = features["c2"]
        c3 = features["c3"]
        
        # Decoder
        seg_logits = self.decoder([c1, c2, c3])
        
        # Upsample segmentation to input size
        if seg_logits.size()[2:] != input_size:
            seg_logits = F.interpolate(seg_logits, size=input_size, mode='bilinear', align_corners=False)
        
        # Aux Head
        aux_logits = self.aux_head(c2)
        
        # Heads
        view_logits = self.view_head(c3)

        return {
            "seg": seg_logits,
            "presence": None,
            "quality": None,
            "view": view_logits,
            "aux": aux_logits,
        }


class MultiStructureGuidanceNet(nn.Module):
    def __init__(self, pretrained=True, num_structures=3, num_views=3):
        super().__init__()

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        return_nodes = {
            "features.1": "c1",
            "features.3": "c2",
            "features.12": "c3",
        }
        self.encoder = create_feature_extractor(backbone, return_nodes=return_nodes)

        c1_ch = 16
        c2_ch = 24
        c3_ch = 576

        self.decoder = LiteUNetDecoder([c3_ch, c2_ch, c1_ch], out_channels=num_structures)

        self.aux_head = nn.Conv2d(c2_ch, num_structures, kernel_size=1)

        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3_ch, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3_ch, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.camus_quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3_ch, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.view_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3_ch, 128),
            nn.ReLU(),
            nn.Linear(128, num_views),
        )

    def forward(self, x):
        input_size = x.size()[2:]

        features = self.encoder(x)
        c1 = features["c1"]
        c2 = features["c2"]
        c3 = features["c3"]

        seg_logits = self.decoder([c1, c2, c3])
        if seg_logits.size()[2:] != input_size:
            seg_logits = F.interpolate(seg_logits, size=input_size, mode='bilinear', align_corners=False)

        aux_logits = self.aux_head(c2)

        presence_logits = self.presence_head(c3)

        quality_logits = self.quality_head(c3)
        camus_quality_logits = self.camus_quality_head(c3)
        view_logits = self.view_head(c3)

        return {
            "seg": seg_logits,
            "presence": presence_logits,
            "quality": quality_logits,
            "camus_quality": camus_quality_logits,
            "view": view_logits,
            "aux": aux_logits,
        }


def load_state_dict_compat(model: nn.Module, state_dict: dict):
    sd = dict(state_dict)
    if "aux_head.0.weight" in sd and "aux_head.weight" not in sd:
        sd["aux_head.weight"] = sd.pop("aux_head.0.weight")
        sd["aux_head.bias"] = sd.pop("aux_head.0.bias")
    drop_prefixes = (
        "structure_presence_head.",
        "quality_from_presence.",
    )
    for k in list(sd.keys()):
        if k.startswith(drop_prefixes):
            sd.pop(k, None)
    return model.load_state_dict(sd, strict=False)

if __name__ == "__main__": 
    # Test the model
    model = CardiacGuidanceNet()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(f"Seg: {out['seg'].shape}, Pres: {out['presence']}, View: {out['view'].shape}")
