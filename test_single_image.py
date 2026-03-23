import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use('Agg')

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.model import MultiStructureGuidanceNet
try:
    from src.model import load_state_dict_compat
except Exception:
    load_state_dict_compat = None
from src.presence import PresenceEvaluator, presence_to_confidence_pct
import albumentations as A
from albumentations.pytorch import ToTensorV2

def test_image(image_path, model_path):
    # Load Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = MultiStructureGuidanceNet(pretrained=False, num_structures=3, num_views=2).to(device)
    try:
        state = torch.load(model_path, map_location=device)
        if load_state_dict_compat is not None:
            missing, unexpected = load_state_dict_compat(model, state)
        else:
            missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"Loaded with missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()
    
    # Load Image
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Failed to load image: {image_path}")
        return
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    augmented = transform(image=img_rgb)["image"]
    img_tensor = augmented.unsqueeze(0).to(device) # (1, 3, 256, 256)
    
    # Inference
    with torch.no_grad():
        out = model(img_tensor)
        seg_probs_t = torch.sigmoid(out["seg"])
        quality_prob_t = torch.sigmoid(out["quality"]) 
        view_logits = out["view"]

        seg_probs = seg_probs_t.cpu().numpy()[0]  # (3, 256, 256)
        quality_score = float(quality_prob_t.item())

        view_probs = torch.softmax(view_logits, dim=1).cpu().numpy()[0]
        view_class = int(np.argmax(view_probs))
        view_score = float(view_probs[view_class])

    presence_eval = PresenceEvaluator()
    pres_info = presence_eval(seg_probs)
    heart_confidence = float(1.0 - (1.0 - pres_info["presence_score"]) * (1.0 - quality_score))
        
    seg_union = np.max(seg_probs, axis=0)

    h, w = seg_union.shape
    img_center = (w/2, h/2)
    
    # Coordinate grid
    ys, xs = np.mgrid[0:h, 0:w]
    
    # Weighted moments
    mass = seg_union.sum()
    
    centroid = None
    dist_px = float('inf')
    q_center = 0.0
    
    # Use a soft threshold for mass to avoid division by zero
    if mass > 1.0:
        cx = (xs * seg_union).sum() / mass
        cy = (ys * seg_union).sum() / mass
        centroid = (cx, cy)
        
        # Distance
        dx = cx - img_center[0]
        dy = cy - img_center[1]
        dist_px = np.sqrt(dx**2 + dy**2)
        
        # Centeredness
        diag_px = np.sqrt(w**2 + h**2)
        q_center = max(0.0, 1.0 - dist_px / diag_px)
        
    q_guidance = heart_confidence * view_score * q_center
    
    # Visualization
    vis_img = cv2.resize(img_orig, (256, 256))
    
    mask_binary = (seg_union > 0.5).astype(np.uint8)

    colored_mask = np.zeros_like(vis_img)
    colored_mask[mask_binary == 1] = [0, 255, 0]
    
    # Blend
    mask_overlay = cv2.addWeighted(vis_img, 1, colored_mask, 0.5, 0)
    
    # Draw Centroid and Center on top
    cv2.circle(mask_overlay, (int(img_center[0]), int(img_center[1])), 4, (255, 0, 0), -1) # Blue center
    
    # Gate arrow drawing
    if centroid and q_guidance > 0.2:
        cv2.circle(mask_overlay, (int(centroid[0]), int(centroid[1])), 6, (0, 0, 255), -1) # Red centroid
        cv2.arrowedLine(mask_overlay, (int(img_center[0]), int(img_center[1])), 
                        (int(centroid[0]), int(centroid[1])), (0, 255, 255), 2) # Yellow arrow
    
    # Annotate
    view_names = ["A4C", "A2C"]
    
    # Text block
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    
    lines = [
        f"View: {view_names[view_class]} ({view_score:.2f})",
        f"Heart: {heart_confidence:.2f} | Q: {quality_score:.2f}",
        f"Dist: {dist_px:.1f}px | Q_cnt: {q_center:.2f}",
        f"Guidance: {q_guidance:.2f}"
    ]
    
    y = 20
    for line in lines:
        cv2.putText(mask_overlay, line, (10, y), font, scale, (0,0,0), thickness+2) # outline
        cv2.putText(mask_overlay, line, (10, y), font, scale, color, thickness)
        y += 20
        
    # Save
    out_dir = os.path.dirname(image_path)
    out_name = os.path.splitext(os.path.basename(image_path))[0] + "_result.png"
    out_path = os.path.join(out_dir, out_name)
    cv2.imwrite(out_path, mask_overlay)
    
    # Console Output
    print("\n" + "="*30)
    print(" INFERENCE RESULTS")
    print("="*30)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Model: {os.path.basename(model_path)}")
    print("-" * 30)
    print(f"Heart Confidence: {heart_confidence:.4f}")
    pres_pct = presence_to_confidence_pct(pres_info["presence_score"])
    print(f"PresenceScore: {pres_info['presence_score']:.4f} ({pres_pct:.1f}%) | LVGeom: {pres_info['lv_geometry']:.0f} | LVAreaRatio: {pres_info['lv_area_ratio']:.4f}")
    print(f"Quality Score: {quality_score:.4f}")
    print(f"View Class: {view_names[view_class]} (Q_view: {view_score:.4f})")
    print("-" * 30)
    print(f"Centroid: {centroid if centroid else 'Not detected'}")
    print(f"Image Center: {img_center}")
    print(f"Distance: {dist_px:.2f} pixels")
    print(f"Centeredness Score (Q_center): {q_center:.4f}")
    print("-" * 30)
    print(f"OVERALL GUIDANCE CONFIDENCE: {q_guidance:.4f}")
    print("="*30)
    print(f"Visualization saved to: {out_path}")

if __name__ == "__main__":
    import sys
    img_path = "/home/tc115/Yue/Ultraprobe_guiding_system/test/test3.png"
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        
    model_path = "checkpoints/best_model_multi.pth"
    test_image(img_path, model_path)
