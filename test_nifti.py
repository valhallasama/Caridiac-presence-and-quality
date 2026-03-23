import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use('Agg')

import torch
import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from src.model import MultiStructureGuidanceNet
try:
    from src.model import load_state_dict_compat
except Exception:
    load_state_dict_compat = None
from src.presence import PresenceEvaluator, TemporalPresenceFilter, presence_to_confidence_pct
import albumentations as A
from albumentations.pytorch import ToTensorV2

def test_nifti(nifti_path, model_path):
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
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()
    
    print(f"Loading NIfTI: {nifti_path}")
    try:
        img_nii = nib.load(nifti_path)
        img_data = img_nii.get_fdata()
        print(f"Original Data Shape: {img_data.shape}")
        print(f"Original Data Range: {img_data.min()} - {img_data.max()}")
    except Exception as e:
        print(f"Failed to load NIfTI: {e}")
        return

    # Preprocess
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    def infer_single_frame(img_2d):
        img_norm = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

        augmented = transform(image=img_rgb)["image"]
        img_tensor = augmented.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_tensor)
            seg_probs_t = torch.sigmoid(out["seg"])
            quality_prob_t = torch.sigmoid(out["quality"])
            view_logits = out["view"]

        seg_probs = seg_probs_t.cpu().numpy()[0]
        quality_score = float(quality_prob_t.item())

        view_probs = torch.softmax(view_logits, dim=1).cpu().numpy()[0]
        view_class = int(np.argmax(view_probs))
        view_score = float(view_probs[view_class])

        presence_eval = PresenceEvaluator()
        pres_info = presence_eval(seg_probs)
        heart_confidence = float(1.0 - (1.0 - pres_info["presence_score"]) * (1.0 - quality_score))

        return {
            "heart_confidence": heart_confidence,
            "presence_score": pres_info["presence_score"],
            "lv_area_ratio": pres_info["lv_area_ratio"],
            "quality": quality_score,
            "view_class": view_class,
            "view_score": view_score,
            "seg_union": np.max(seg_probs, axis=0),
        }

    if img_data.ndim == 3 and img_data.shape[-1] > 1:
        t = int(img_data.shape[-1])
        presence_raw = []
        presence_ema = []
        presence_final = []
        views = []
        qs = []

        presence_filter = TemporalPresenceFilter(ema_alpha=0.3, window=40, lv_var_target=0.05, hold_decay=0.97)

        for frame_idx in range(t):
            out = infer_single_frame(img_data[:, :, frame_idx])
            presence_raw.append(out["presence_score"])
            qs.append(out["quality"])
            views.append(out["view_class"])

            tf = presence_filter.update(out["presence_score"], out["lv_area_ratio"])
            presence_ema.append(tf["presence_ema"])
            presence_final.append(tf["presence_final"])

        presence_raw = np.array(presence_raw, dtype=np.float32)
        presence_ema = np.array(presence_ema, dtype=np.float32)
        presence_final = np.array(presence_final, dtype=np.float32)
        qs = np.array(qs, dtype=np.float32)
        views = np.array(views, dtype=np.int64)

        out_dir = os.path.dirname(nifti_path)
        base = os.path.basename(nifti_path)
        if base.endswith('.nii.gz'):
            base = base.replace('.nii.gz', '')
        elif base.endswith('.nii'):
            base = base.replace('.nii', '')

        raw_pct = np.array([presence_to_confidence_pct(v) for v in presence_raw.tolist()], dtype=np.float32)
        ema_pct = np.array([presence_to_confidence_pct(v) for v in presence_ema.tolist()], dtype=np.float32)
        final_pct = np.array([presence_to_confidence_pct(v) for v in presence_final.tolist()], dtype=np.float32)

        npz_path = os.path.join(out_dir, base + "_scores.npz")
        np.savez(
            npz_path,
            raw_presence_score=presence_raw,
            ema_presence_score=presence_ema,
            presence_final=presence_final,
            raw_confidence_pct=raw_pct,
            ema_confidence_pct=ema_pct,
            final_confidence_pct=final_pct,
            quality=qs,
            view_class=views,
        )

        plt.figure(figsize=(10, 3))
        plt.plot(raw_pct, label='raw%')
        plt.plot(ema_pct, label='ema%')
        plt.plot(final_pct, label='final%')
        plt.ylim(0, 100)
        plt.xlabel('frame')
        plt.ylabel('presence confidence (%)')
        plt.legend()
        plot_path = os.path.join(out_dir, base + "_scores.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print("\n" + "="*30)
        print(" SEQUENCE INFERENCE RESULTS")
        print("="*30)
        print(f"Frames: {t}")
        print(f"Raw Presence: mean={presence_raw.mean():.4f} min={presence_raw.min():.4f} max={presence_raw.max():.4f}")
        print(f"EMA Presence: mean={presence_ema.mean():.4f} min={presence_ema.min():.4f} max={presence_ema.max():.4f}")
        print(f"Final Presence: mean={presence_final.mean():.4f} min={presence_final.min():.4f} max={presence_final.max():.4f}")
        print(f"Final Confidence: mean={final_pct.mean():.1f}% min={final_pct.min():.1f}% max={final_pct.max():.1f}%")
        print(f"Saved scores: {npz_path}")
        print(f"Saved plot: {plot_path}")

        thr = 0.10
        present = presence_final > thr
        segments = []
        start = None
        for i, v in enumerate(present.tolist()):
            if v and start is None:
                start = i
            if (not v) and start is not None:
                segments.append((start, i - 1))
                start = None
        if start is not None:
            segments.append((start, len(present) - 1))

        total_present = int(present.sum())
        print("\n" + "=" * 30)
        print(" BALANCED PRESENCE (final > 0.10)")
        print("=" * 30)
        print(f"Frames above thr: {total_present}/{len(present)} ({100.0 * float(total_present) / float(len(present)):.2f}%)")
        print(f"Segments: {len(segments)}")
        if len(segments) > 0:
            print("First 5 segments:")
            for s, e in segments[:5]:
                print(f"  {s}-{e}")
        return
    
    if img_data.ndim == 3:
        img_data = img_data[:, :, 0]

    img_norm = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)
    img_uint8 = (img_norm * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

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

    structure_names = ["LV", "Myo", "LA"]
    presence_eval = PresenceEvaluator()
    pres_info = presence_eval(seg_probs)
    heart_confidence = float(1.0 - (1.0 - pres_info["presence_score"]) * (1.0 - quality_score))

    seg_union = np.max(seg_probs, axis=0)
        
    print(f"Seg Union Stats - Min: {seg_union.min():.4f}, Max: {seg_union.max():.4f}, Mean: {seg_union.mean():.4f}")
    
    # Dynamic Thresholding for Visualization
    # If the model is not confident (range is small), we normalize for vis
    mask_min = seg_union.min()
    mask_max = seg_union.max()
    mask_range = mask_max - mask_min
    
    print(f"Dynamic Thresholding: Range [{mask_min:.4f}, {mask_max:.4f}]")
    
    if mask_range < 0.3: # Low contrast or weak prediction
        print("Using relative thresholding...")
        # Threshold at half the range above min
        threshold = mask_min + (mask_range * 0.5)
        mask_binary = (seg_union > threshold).astype(np.uint8)
    else:
        mask_binary = (seg_union > 0.5).astype(np.uint8)
    
    # 2. Soft Centroid
    h, w = seg_union.shape
    img_center = (w/2, h/2)
    
    ys, xs = np.mgrid[0:h, 0:w]
    mass = seg_union.sum()
    
    centroid = None
    dist_px = float('inf')
    q_center = 0.0
    
    if mass > 1.0:
        cx = (xs * seg_union).sum() / mass
        cy = (ys * seg_union).sum() / mass
        centroid = (cx, cy)
        
        dx = cx - img_center[0]
        dy = cy - img_center[1]
        dist_px = np.sqrt(dx**2 + dy**2)
        
        diag_px = np.sqrt(w**2 + h**2)
        q_center = max(0.0, 1.0 - dist_px / diag_px) 
        
    q_guidance = heart_confidence * view_score * q_center
    
    # Visualization
    # Use the uint8 image for visualization
    vis_img = cv2.resize(img_rgb, (256, 256))
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR) # Back to BGR for OpenCV
    
    mask_binary = (seg_union > 0.5).astype(np.uint8)
    colored_mask = np.zeros_like(vis_img)
    colored_mask[mask_binary == 1] = [0, 255, 0] 
    
    mask_overlay = cv2.addWeighted(vis_img, 1, colored_mask, 0.5, 0)
    
    # Generate Heatmap for side-by-side view
    heatmap = (seg_union * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Draw Indicators
    cv2.circle(mask_overlay, (int(img_center[0]), int(img_center[1])), 4, (255, 0, 0), -1) 
    if centroid and q_guidance >= 0.2:
        cv2.circle(mask_overlay, (int(centroid[0]), int(centroid[1])), 6, (0, 0, 255), -1) 
        cv2.arrowedLine(mask_overlay, (int(img_center[0]), int(img_center[1])), 
                        (int(centroid[0]), int(centroid[1])), (0, 255, 255), 2)
    
    view_names = ["A4C", "A2C"]
    
    lines = [
        f"View: {view_names[view_class]} ({view_score:.2f})",
        f"Heart: {heart_confidence:.2f} | Q: {quality_score:.2f}",
        f"Dist: {dist_px:.1f}px | Q_cnt: {q_center:.2f}",
        f"Guidance: {q_guidance:.2f}"
    ]
    
    y = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    for line in lines:
        cv2.putText(mask_overlay, line, (10, y), font, 0.5, (0,0,0), 3) 
        cv2.putText(mask_overlay, line, (10, y), font, 0.5, (255, 255, 255), 1)
        y += 20
        
    # Save
    out_dir = os.path.dirname(nifti_path)
    out_name = os.path.splitext(os.path.basename(nifti_path))[0] + "_result.png"
    if nifti_path.endswith('.nii.gz'):
        out_name = os.path.basename(nifti_path).replace('.nii.gz', '_result.png')
    elif nifti_path.endswith('.nii'):
        out_name = os.path.basename(nifti_path).replace('.nii', '_result.png')
        
    out_path = os.path.join(out_dir, out_name)
    # Re-stack for final output to include both
    final_output = np.hstack((mask_overlay, heatmap_color))
    cv2.imwrite(out_path, final_output)
    
    print("\n" + "="*30)
    print(" NIFTI INFERENCE RESULTS")
    print("="*30)
    print(f"File: {os.path.basename(nifti_path)}")
    print(f"View Class: {view_names[view_class]} ({view_score:.4f})")
    print(f"Heart Confidence: {heart_confidence:.4f}")
    for i, name in enumerate(structure_names):
        print(f"{name} Presence: {struct_pres[i]:.4f} | {name} Q_seg: {q_seg_struct[i]:.4f} | {name} Evidence: {evidence_struct[i]:.4f}")
    print(f"Quality Score: {quality_score:.4f}")
    print(f"Centroid: {centroid}")
    print(f"Guidance: {q_guidance:.4f}")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    import sys
    nifti_path = "/home/tc115/Yue/Ultraprobe_guiding_system/data/CAMUS_public/database_nifti/patient0001/patient0001_2CH_ED.nii.gz"
    if len(sys.argv) > 1:
        nifti_path = sys.argv[1]
    if not os.path.exists(nifti_path) and nifti_path.endswith('.nii.gz'):
        alt = nifti_path[:-3]
        if os.path.exists(alt):
            nifti_path = alt

    model_path = "checkpoints/best_model_multi.pth"
    if len(sys.argv) > 2:
        model_path = sys.argv[2]

    test_nifti(nifti_path, model_path)
