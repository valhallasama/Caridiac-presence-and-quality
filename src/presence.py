import numpy as np
import cv2


class PresenceEvaluator:
    def __init__(
        self,
        structure_weights=(0.5, 0.2, 0.3),
        expected_area_ratios=(0.15, 0.25, 0.10),
        bin_thresh=0.5,
        lv_area_ratio_min=0.002,
        lv_area_ratio_max=0.35,
        lv_largest_cc_ratio_min=0.7,
        lv_max_components=3,
        aspect_ratio_min=0.2,
        aspect_ratio_max=5.0,
    ):
        self.structure_weights = np.asarray(structure_weights, dtype=np.float32)
        self.expected_area_ratios = np.asarray(expected_area_ratios, dtype=np.float32)
        self.bin_thresh = float(bin_thresh)
        self.lv_area_ratio_min = float(lv_area_ratio_min)
        self.lv_area_ratio_max = float(lv_area_ratio_max)
        self.lv_largest_cc_ratio_min = float(lv_largest_cc_ratio_min)
        self.lv_max_components = int(lv_max_components)
        self.aspect_ratio_min = float(aspect_ratio_min)
        self.aspect_ratio_max = float(aspect_ratio_max)

    def _lv_geometry_valid(self, lv_prob_2d: np.ndarray, valid_mask_hw: np.ndarray | None = None):
        h, w = lv_prob_2d.shape
        if valid_mask_hw is not None:
            vm = (np.asarray(valid_mask_hw) > 0).astype(np.uint8)
            lv_prob_2d = lv_prob_2d * vm
            valid_frac = float(vm.mean())
        else:
            vm = None
            valid_frac = 1.0

        bin_mask = (lv_prob_2d > self.bin_thresh).astype(np.uint8)
        area = int(bin_mask.sum())
        area_ratio = float(area) / float(h * w)

        min_a = self.lv_area_ratio_min
        max_a = self.lv_area_ratio_max
        if area_ratio < min_a or area_ratio > max_a:
            return 0.0, area_ratio

        if vm is not None:
            bin_mask = (bin_mask & vm).astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(bin_mask, connectivity=8)
        if num_labels <= 1:
            return 0.0, area_ratio

        comp_areas = np.bincount(labels.reshape(-1))
        comp_areas[0] = 0

        comp_count = int((comp_areas > 0).sum())
        if comp_count > self.lv_max_components:
            return 0.0, area_ratio

        largest = int(comp_areas.max())
        if area <= 0:
            return 0.0, area_ratio

        largest_ratio = float(largest) / float(area)
        if largest_ratio < self.lv_largest_cc_ratio_min:
            return 0.0, area_ratio

        largest_label = int(comp_areas.argmax())
        ys, xs = np.where(labels == largest_label)
        if len(xs) < 5:
            return 0.0, area_ratio

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw = max(1, x1 - x0 + 1)
        bh = max(1, y1 - y0 + 1)
        aspect = float(bw) / float(bh)

        if aspect < self.aspect_ratio_min or aspect > self.aspect_ratio_max:
            return 0.0, area_ratio

        return 1.0, area_ratio

    def __call__(self, seg_probs_chw: np.ndarray, valid_mask_hw: np.ndarray | None = None):
        seg_probs = np.asarray(seg_probs_chw, dtype=np.float32)
        if seg_probs.ndim != 3:
            raise ValueError("seg_probs must be (C,H,W)")

        if valid_mask_hw is not None:
            vm = (np.asarray(valid_mask_hw) > 0).astype(np.float32)
            valid_frac = float(vm.mean())
            seg_m = seg_probs * vm[None, :, :]
            area = seg_m.mean(axis=(1, 2))
            peak = seg_m.max(axis=(1, 2))
        else:
            vm = None
            valid_frac = 1.0
            area = seg_probs.mean(axis=(1, 2))
            peak = seg_probs.max(axis=(1, 2))

        exp = self.expected_area_ratios * valid_frac
        if len(exp) != seg_probs.shape[0]:
            exp = np.ones((seg_probs.shape[0],), dtype=np.float32) * 0.15 * valid_frac

        area_norm = np.clip(area / (exp + 1e-8), 0.0, 1.0)
        peak_gate = np.clip((peak - 0.5) / 0.5, 0.0, 1.0)
        s = area_norm * peak_gate
        w = self.structure_weights
        if len(w) != seg_probs.shape[0]:
            w = np.ones((seg_probs.shape[0],), dtype=np.float32) / float(seg_probs.shape[0])

        structure_score = float(np.clip((w * s).sum(), 0.0, 1.0))

        if vm is not None:
            lv_geom, lv_area_ratio = self._lv_geometry_valid(seg_probs[0], vm)
        else:
            lv_geom, lv_area_ratio = self._lv_geometry_valid(seg_probs[0], None)

        coexist = float(min(area_norm[0], area_norm[2])) if seg_probs.shape[0] >= 3 else float(area_norm[0])
        agreement = float(np.clip(coexist, 0.0, 1.0))

        presence_score = float(np.clip(structure_score * lv_geom * (0.25 + 0.75 * agreement), 0.0, 1.0))

        return {
            "presence_score": presence_score,
            "structure_score": structure_score,
            "lv_geometry": float(lv_geom),
            "lv_area_ratio": float(lv_area_ratio),
            "area": area,
            "peak": peak,
        }


class TemporalPresenceFilter:
    def __init__(self, ema_alpha=0.3, window=40, lv_var_target=0.05, hold_decay=0.97):
        self.ema_alpha = float(ema_alpha)
        self.window = int(window)
        self.lv_var_target = float(lv_var_target)
        self.hold_decay = float(hold_decay)
        self._ema = None
        self._final = None
        self._lv_area = []

    def update(self, presence_score: float, lv_area_ratio: float):
        ps = float(presence_score)
        lv = float(lv_area_ratio)

        if self._ema is None:
            self._ema = ps
        else:
            self._ema = (1.0 - self.ema_alpha) * self._ema + self.ema_alpha * ps

        self._lv_area.append(lv)
        if len(self._lv_area) > self.window:
            self._lv_area = self._lv_area[-self.window :]

        temporal = 0.0
        if len(self._lv_area) >= max(5, self.window // 2):
            std = float(np.std(self._lv_area))
            temporal = float(np.clip(std / (self.lv_var_target + 1e-8), 0.0, 1.0))

        # Base confidence: EMA modulated by temporal evidence, but with a milder factor
        base = float(np.clip(self._ema * (0.4 + 0.6 * temporal), 0.0, 1.0))

        # Presence hold: fast rise, slow decay to reduce jitter when heart is present.
        if self._final is None:
            self._final = base
        else:
            if base >= self._final:
                # Allow presence to rise immediately when evidence increases.
                self._final = base
            else:
                # When evidence drops, decay slowly instead of following every dip.
                self._final = max(base, self._final * self.hold_decay)

        presence_final = float(np.clip(self._final, 0.0, 1.0))

        return {
            "presence_ema": float(self._ema),
            "temporal": temporal,
            "presence_final": presence_final,
        }


def presence_to_confidence_pct(presence_score: float) -> float:
    p = float(presence_score)
    p = float(np.clip(p, 0.0, 1.0))
    xs = np.array([0.0, 0.05, 0.10, 0.20, 0.40, 1.0], dtype=np.float32)
    ys = np.array([0.0, 0.0, 10.0, 40.0, 90.0, 99.0], dtype=np.float32)
    return float(np.interp(p, xs, ys))


def quality_to_confidence_pct(quality_score: float) -> float:
    q = float(quality_score)
    q = float(np.clip(q, 0.0, 1.0))
    xs = np.array([0.0, 0.3, 0.6, 0.9, 1.0], dtype=np.float32)
    ys = np.array([0.0, 10.0, 30.0, 60.0, 99.0], dtype=np.float32)
    return float(np.interp(q, xs, ys))
