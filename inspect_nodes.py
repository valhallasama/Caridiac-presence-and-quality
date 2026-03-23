import torch
from torchvision import models
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

def inspect_nodes():
    model = models.mobilenet_v3_small(pretrained=True)
    nodes, _ = get_graph_node_names(model)
    print("Graph Nodes:")
    for node in nodes:
        if "features" in node:
            print(node)
            
    # Let's try to extract specific ones and check shapes
    # User suggested: features.1 (1/4), features.3 (1/8), features.12 (1/16)
    # Note: MobileNetV3Small stride locations:
    # layer 0: stride 2 (1/2) -> features.0
    # layer 1: stride 2 (1/4) -> features.1.block.0 ?? No, let's check.
    
    return_nodes = {
        "features.0": "c0",
        "features.1": "c1", 
        "features.2": "c2_pre",
        "features.3": "c2",
        "features.8": "c3_pre",
        "features.12": "c3"
    }
    
    # We might need to adjust based on actual graph
    try:
        extractor = create_feature_extractor(model, return_nodes=return_nodes)
        x = torch.randn(1, 3, 256, 256)
        out = extractor(x)
        print("\nShapes:")
        for k, v in out.items():
            print(f"{k}: {v.shape}")
    except ValueError as e:
        print(f"\nError creating extractor: {e}")
        # Fallback to print all nodes to debug
        # print(nodes)

if __name__ == "__main__":
    inspect_nodes()
