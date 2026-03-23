import torch
from torchvision import models
from src.model import CardiacGuidanceNet

def inspect():
    model = models.mobilenet_v3_small(pretrained=False)
    print("MobileNetV3 Small Structure:")
    # print(model.features)
    
    # Check our slicing
    features = model.features
    c1_part = features[:2]
    c2_part = features[2:9]
    c3_part = features[9:]
    
    x = torch.randn(1, 3, 256, 256)
    
    out1 = c1_part(x)
    print(f"C1 (Layer 0-1) output: {out1.shape}")
    
    out2 = c2_part(out1)
    print(f"C2 (Layer 2-8) output: {out2.shape}")
    
    out3 = c3_part(out2)
    print(f"C3 (Layer 9-end) output: {out3.shape}")
    
    # Check full model
    full_model = CardiacGuidanceNet(pretrained=False)
    out = full_model(x)
    print(f"\nFull Model Output:")
    print(f"Seg: {out['seg'].shape}")
    print(f"Presence: {out['presence']}")
    print(f"View: {out['view'].shape}")

if __name__ == "__main__":
    inspect()
