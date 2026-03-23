import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model import MultiStructureGuidanceNet
try:
    from src.model import load_state_dict_compat
except Exception:
    load_state_dict_compat = None

from src.presence import PresenceEvaluator, TemporalPresenceFilter, presence_to_confidence_pct, quality_to_confidence_pct

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _compute_fan_mask_bgr(frame_bgr_256: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr_256, cv2.COLOR_BGR2GRAY)
    # Ultrasound wedge is typically non-black; use a low threshold to keep dark tissue speckle.
    thr = 5
    _, m = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    num_labels, labels = cv2.connectedComponents(m, connectivity=8)
    if num_labels <= 1:
        return np.ones((frame_bgr_256.shape[0], frame_bgr_256.shape[1]), dtype=np.uint8)

    areas = np.bincount(labels.reshape(-1))
    areas[0] = 0
    lab = int(np.argmax(areas))
    out = (labels == lab).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_DILATE, np.ones((9, 9), np.uint8), iterations=1)
    return out


def _largest_cc(mask_u8: np.ndarray) -> np.ndarray:
    m = (np.asarray(mask_u8) > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(m, connectivity=8)
    if num_labels <= 1:
        return m
    areas = np.bincount(labels.reshape(-1))
    areas[0] = 0
    lab = int(np.argmax(areas))
    return (labels == lab).astype(np.uint8)


def infer_single_frame_bgr(frame_bgr, model, device, transform, presence_eval: PresenceEvaluator):
    """Infer presence and auxiliary outputs for a single BGR frame.

    Returns a dict with keys:
      - presence_score
      - lv_area_ratio
      - quality
      - view_class
      - view_score
      - overlay_bgr: 256x256 BGR frame with segmentation overlay
      - base_bgr: 256x256 resized original frame
    """
    if frame_bgr is None:
        return None

    vis_img = cv2.resize(frame_bgr, (256, 256))
    fan_mask_u8 = _compute_fan_mask_bgr(vis_img)
    masked_bgr = vis_img.copy()
    masked_bgr[fan_mask_u8 == 0] = 0
    img_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)

    augmented = transform(image=img_rgb)["image"]
    img_tensor = augmented.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_tensor)
        seg_probs_t = torch.sigmoid(out["seg"])          # (1, C, H, W)
        quality_prob_t = torch.sigmoid(out["quality"])  # (1, 1)
        camus_quality_prob_t = torch.sigmoid(out.get("camus_quality", quality_prob_t))  # (1, 1)
        view_logits = out["view"]                       # (1, num_views)

    seg_probs = seg_probs_t.cpu().numpy()[0]          # (C, H, W)
    quality_score = float(quality_prob_t.item())
    camus_quality_score = float(camus_quality_prob_t.item())

    view_probs = torch.softmax(view_logits, dim=1).cpu().numpy()[0]
    view_class = int(np.argmax(view_probs))
    view_score = float(view_probs[view_class])

    pres_info = presence_eval(seg_probs, fan_mask_u8)

    fan_mask = fan_mask_u8.astype(np.float32)
    seg_probs_vis = seg_probs * fan_mask[None, :, :]

    colored_mask = np.zeros_like(vis_img)

    # Simple color scheme: LV=red, Myo=green, LA=blue
    colors = [
        (0, 0, 255),   # LV
        (0, 255, 0),   # Myo
        (255, 0, 0),   # LA
    ]
    pmax = seg_probs_vis.max(axis=0)
    cmax = seg_probs_vis.argmax(axis=0)
    for c in range(min(seg_probs.shape[0], len(colors))):
        mask = ((cmax == c) & (pmax > 0.65)).astype(np.uint8)
        mask = _largest_cc(mask)
        if mask.sum() == 0:
            continue
        col = colors[c]
        for k in range(3):
            colored_mask[:, :, k][mask == 1] = col[k]

    overlay = cv2.addWeighted(vis_img, 1.0, colored_mask, 0.5, 0.0)

    return {
        "presence_score": pres_info["presence_score"],
        "camus_quality": camus_quality_score,
        "lv_area_ratio": pres_info["lv_area_ratio"],
        "lv_geometry": pres_info["lv_geometry"],
        "quality": quality_score,
        "view_class": view_class,
        "view_score": view_score,
        "overlay_bgr": overlay,
        "base_bgr": vis_img,
    }


def test_video(video_path, model_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if model_path is None:
        model_path = os.path.join("checkpoints", "best_model_v2.pth")

    if not os.path.exists(model_path):
        print(f"Model checkpoint not found: {model_path}")
        return

    model = MultiStructureGuidanceNet(pretrained=False, num_structures=3, num_views=2).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Handle both full checkpoint and direct state_dict formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'best_val_dice' in checkpoint:
                print(f"Checkpoint Dice: {checkpoint['best_val_dice']:.4f}")
        else:
            state = checkpoint
        
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

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path}")
    print(f"Frames (reported): {frame_count}, FPS: {fps:.2f}")

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    presence_raw = []
    presence_ema = []
    presence_final = []
    qs = []
    qs_ema = []
    camus_qs = []
    camus_qs_ema = []
    views = []

    presence_eval = PresenceEvaluator()
    presence_filter = TemporalPresenceFilter(ema_alpha=0.3, window=40, lv_var_target=0.05, hold_decay=0.97)

    quality_ema = None
    camus_quality_ema = None
    quality_alpha = 0.3
    quality_gate_pct = 50.0
    quality_prev_on = False

    # Prepare video writer for overlay output (256x256)
    out_dir = os.path.dirname(video_path)
    base = os.path.splitext(os.path.basename(video_path))[0]
    overlay_path = os.path.join(out_dir, base + "_seg_overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_out = fps if fps and fps > 0 else 30.0
    writer = cv2.VideoWriter(overlay_path, fourcc, fps_out, (256, 256))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out = infer_single_frame_bgr(frame, model, device, transform, presence_eval)
        if out is None:
            break

        ps = float(out["presence_score"])
        presence_raw.append(ps)
        tf = presence_filter.update(ps, out["lv_area_ratio"])
        presence_ema.append(tf["presence_ema"])
        presence_final.append(tf["presence_final"])
        conf_pct = presence_to_confidence_pct(tf["presence_final"])

        quality_on = conf_pct >= float(quality_gate_pct)
        if quality_on:
            if quality_ema is None:
                quality_ema = out["quality"]
            else:
                quality_ema = quality_alpha * out["quality"] + (1.0 - quality_alpha) * quality_ema
            qs.append(float(out["quality"]))
            qs_ema.append(float(quality_ema))
            
            # CAMUS quality tracking
            if camus_quality_ema is None:
                camus_quality_ema = out["camus_quality"]
            else:
                camus_quality_ema = quality_alpha * out["camus_quality"] + (1.0 - quality_alpha) * camus_quality_ema
            camus_qs.append(float(out["camus_quality"]))
            camus_qs_ema.append(float(camus_quality_ema))
        else:
            qs.append(0.0)
            quality_ema = 0.0
            qs_ema.append(0.0)
            camus_qs.append(0.0)
            camus_quality_ema = 0.0
            camus_qs_ema.append(0.0)
        quality_prev_on = bool(quality_on)

        views.append(out["view_class"])

        q_ema_pct = float(quality_to_confidence_pct(quality_ema))
        show_overlay = conf_pct >= 10.0
        frame_out = out["overlay_bgr"] if show_overlay else out["base_bgr"]

        if bool(quality_on):
            text = f"Pres {conf_pct:.0f}% | Qual {q_ema_pct:.0f}%"
        else:
            text = f"Pres {conf_pct:.0f}%"
        cv2.putText(frame_out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        writer.write(frame_out)

        frame_idx += 1

    cap.release()
    writer.release()

    if len(presence_raw) == 0:
        print("No frames processed.")
        return

    presence_raw = np.array(presence_raw, dtype=np.float32)
    presence_ema = np.array(presence_ema, dtype=np.float32)
    presence_final = np.array(presence_final, dtype=np.float32)
    qs = np.array(qs, dtype=np.float32)
    qs_ema = np.array(qs_ema, dtype=np.float32)
    camus_qs = np.array(camus_qs, dtype=np.float32)
    camus_qs_ema = np.array(camus_qs_ema, dtype=np.float32)
    views = np.array(views, dtype=np.int64)

    raw_pct = np.array([presence_to_confidence_pct(v) for v in presence_raw.tolist()], dtype=np.float32)
    ema_pct = np.array([presence_to_confidence_pct(v) for v in presence_ema.tolist()], dtype=np.float32)
    final_pct = np.array([presence_to_confidence_pct(v) for v in presence_final.tolist()], dtype=np.float32)
    quality_pct = np.array([quality_to_confidence_pct(float(v)) for v in qs.tolist()], dtype=np.float32)
    quality_ema_pct = np.array([quality_to_confidence_pct(float(v)) for v in qs_ema.tolist()], dtype=np.float32)
    camus_quality_pct = np.array([quality_to_confidence_pct(float(v)) for v in camus_qs.tolist()], dtype=np.float32)
    camus_quality_ema_pct = np.array([quality_to_confidence_pct(float(v)) for v in camus_qs_ema.tolist()], dtype=np.float32)

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
        quality_ema=qs_ema,
        quality_pct=quality_pct,
        quality_ema_pct=quality_ema_pct,
        camus_quality=camus_qs,
        camus_quality_ema=camus_qs_ema,
        camus_quality_pct=camus_quality_pct,
        camus_quality_ema_pct=camus_quality_ema_pct,
        view_class=views,
    )

    plt.figure(figsize=(12, 4))
    plt.plot(raw_pct, label="raw%", alpha=0.5)
    plt.plot(ema_pct, label="ema%", alpha=0.7)
    plt.plot(final_pct, label="final%", linewidth=2)
    plt.plot(quality_ema_pct, label="quality%", linestyle='--')
    plt.plot(camus_quality_ema_pct, label="camus_quality%", linestyle=':')
    plt.ylim(0, 100)
    plt.xlabel("frame")
    plt.ylabel("score (%)")
    plt.legend()
    plt.title("Presence and Quality Scores")
    plot_path = os.path.join(out_dir, base + "_scores.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\n" + "=" * 30)
    print(" VIDEO INFERENCE RESULTS")
    print("=" * 30)
    print(f"Frames processed: {len(presence_raw)}")
    print(f"Raw Presence: mean={presence_raw.mean():.4f} min={presence_raw.min():.4f} max={presence_raw.max():.4f}")
    print(f"EMA Presence: mean={presence_ema.mean():.4f} min={presence_ema.min():.4f} max={presence_ema.max():.4f}")
    print(f"Final Presence: mean={presence_final.mean():.4f} min={presence_final.min():.4f} max={presence_final.max():.4f}")
    print(f"Final Confidence: mean={final_pct.mean():.1f}% min={final_pct.min():.1f}% max={final_pct.max():.1f}%")
    print(f"Quality (gated): mean={qs.mean():.4f} min={qs.min():.4f} max={qs.max():.4f}")
    print(f"Quality% (gated): mean={quality_ema_pct.mean():.1f}% min={quality_ema_pct.min():.1f}% max={quality_ema_pct.max():.1f}%")
    print(f"CAMUS Quality (gated): mean={camus_qs.mean():.4f} min={camus_qs.min():.4f} max={camus_qs.max():.4f}")
    print(f"CAMUS Quality% (gated): mean={camus_quality_ema_pct.mean():.1f}% min={camus_quality_ema_pct.min():.1f}% max={camus_quality_ema_pct.max():.1f}%")

    present50 = final_pct >= float(quality_gate_pct)
    if int(present50.sum()) > 0:
        q_present = quality_ema_pct[present50]
        print(f"Quality% when Presence>={quality_gate_pct:.0f}%: mean={float(q_present.mean()):.1f}% min={float(q_present.min()):.1f}% max={float(q_present.max()):.1f}%")
    else:
        print(f"Quality% when Presence>={quality_gate_pct:.0f}%: n/a (no frames)")
    print(f"Saved scores: {npz_path}")
    print(f"Saved plot: {plot_path}")

    thr = 0.012
    present = presence_final > thr

    min_consecutive = 8

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
    longest = 0
    longest_seg = None
    for s, e in segments:
        ln = e - s + 1
        if ln > longest:
            longest = ln
            longest_seg = (s, e)

    print("\n" + "=" * 30)
    print(" BALANCED PRESENCE (final > 0.012)")
    print("=" * 30)
    print(f"Frames above thr: {total_present}/{len(present)} ({100.0 * float(total_present) / float(len(present)):.2f}%)")
    if fps and fps > 0:
        print(f"Total present time: {float(total_present) / float(fps):.2f}s")
    print(f"Segments: {len(segments)}")
    if longest_seg is not None and fps and fps > 0:
        s, e = longest_seg
        print(f"Longest segment: frames {s}-{e} ({float(e - s + 1) / float(fps):.2f}s) @ t={float(s) / float(fps):.2f}-{float(e) / float(fps):.2f}s")
    elif longest_seg is not None:
        s, e = longest_seg
        print(f"Longest segment: frames {s}-{e} ({e - s + 1} frames)")

    if len(segments) > 0:
        print("First 5 segments:")
        for s, e in segments[:5]:
            seg_max = float(final_pct[s : e + 1].max())
            if fps and fps > 0:
                print(f"  {s}-{e}  t={float(s) / float(fps):.2f}-{float(e) / float(fps):.2f}s  max={seg_max:.1f}%")
            else:
                print(f"  {s}-{e}  max={seg_max:.1f}%")

    validated = [(s, e) for (s, e) in segments if (e - s + 1) >= min_consecutive]
    present_valid = np.zeros_like(present, dtype=bool)
    for s, e in validated:
        present_valid[s : e + 1] = True

    total_valid = int(present_valid.sum())
    longest_v = 0
    longest_v_seg = None
    for s, e in validated:
        ln = e - s + 1
        if ln > longest_v:
            longest_v = ln
            longest_v_seg = (s, e)

    print("\n" + "=" * 30)
    print(f" DEBOUNCED PRESENCE (>= {min_consecutive} consecutive frames)")
    print("=" * 30)
    print(f"Frames above thr (debounced): {total_valid}/{len(present)} ({100.0 * float(total_valid) / float(len(present)):.2f}%)")
    if fps and fps > 0:
        print(f"Total present time (debounced): {float(total_valid) / float(fps):.2f}s")
    print(f"Validated segments: {len(validated)}")
    if longest_v_seg is not None and fps and fps > 0:
        s, e = longest_v_seg
        print(f"Longest validated segment: frames {s}-{e} ({float(e - s + 1) / float(fps):.2f}s) @ t={float(s) / float(fps):.2f}-{float(e) / float(fps):.2f}s")
    elif longest_v_seg is not None:
        s, e = longest_v_seg
        print(f"Longest validated segment: frames {s}-{e} ({e - s + 1} frames)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_video.py <video_path> [model_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    test_video(video_path, model_path)
