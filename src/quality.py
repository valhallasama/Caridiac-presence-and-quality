"""
Priority 4: Multi-factor quality calculation
Decouples quality from segmentation-only approach
"""
import numpy as np
import cv2
from scipy.ndimage import sobel


class MultiFactorQualityEvaluator:
    """
    Quality evaluation based on multiple factors, not just segmentation.
    
    Factors:
    1. Edge sharpness (40%) - how clear are the cardiac boundaries
    2. Contrast (30%) - dynamic range within fan region
    3. Segmentation consistency (20%) - how well structures are segmented
    4. Temporal stability (10%) - consistency over time (optional)
    """
    
    def __init__(self):
        self.prev_quality = None
        self.temporal_weight = 0.1
    
    def _compute_edge_sharpness(self, img_gray: np.ndarray, mask: np.ndarray = None) -> float:
        """
        Compute edge sharpness using Sobel gradients.
        Higher gradient magnitude = sharper edges = better quality.
        """
        if img_gray.ndim == 3:
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
        
        # Apply mask if provided
        if mask is not None:
            img_gray = img_gray.astype(np.float32) * (mask > 0).astype(np.float32)
        
        # Compute gradients
        grad_x = sobel(img_gray, axis=1)
        grad_y = sobel(img_gray, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize by valid region
        if mask is not None:
            valid_pixels = (mask > 0).sum()
            if valid_pixels < 100:
                return 0.0
            edge_strength = gradient_magnitude[mask > 0].mean()
        else:
            edge_strength = gradient_magnitude.mean()
        
        # Normalize to 0-1 (empirical range: 0-50 for typical ultrasound)
        sharpness = np.clip(edge_strength / 50.0, 0.0, 1.0)
        
        return float(sharpness)
    
    def _compute_contrast(self, img_gray: np.ndarray, mask: np.ndarray = None) -> float:
        """
        Compute local contrast using standard deviation.
        Higher std = better contrast = better quality.
        """
        if img_gray.ndim == 3:
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
        
        # Apply mask if provided
        if mask is not None:
            valid_pixels = img_gray[mask > 0]
            if len(valid_pixels) < 100:
                return 0.0
        else:
            valid_pixels = img_gray.flatten()
        
        # Compute contrast metrics
        std = valid_pixels.std()
        mean = valid_pixels.mean()
        
        # Coefficient of variation (normalized contrast)
        if mean > 1e-6:
            contrast = std / mean
        else:
            contrast = 0.0
        
        # Normalize to 0-1 (empirical range: 0-1.5 for typical ultrasound)
        contrast = np.clip(contrast / 1.5, 0.0, 1.0)
        
        return float(contrast)
    
    def _compute_segmentation_consistency(self, seg_probs: np.ndarray, mask: np.ndarray = None) -> float:
        """
        Compute how consistent/confident the segmentation is.
        High confidence in predictions = better quality.
        """
        if seg_probs.ndim != 3:
            return 0.0
        
        # Apply mask if provided
        if mask is not None:
            seg_probs = seg_probs * (mask[None, :, :] > 0).astype(np.float32)
        
        # Compute confidence: how many pixels have high probability
        high_conf_pixels = (seg_probs > 0.7).sum()
        total_pixels = seg_probs.shape[1] * seg_probs.shape[2]
        
        if mask is not None:
            total_pixels = (mask > 0).sum()
            if total_pixels < 100:
                return 0.0
        
        # Ratio of high-confidence pixels
        consistency = high_conf_pixels / (total_pixels * seg_probs.shape[0] + 1e-8)
        consistency = np.clip(consistency, 0.0, 1.0)
        
        return float(consistency)
    
    def __call__(self, img_rgb: np.ndarray, seg_probs: np.ndarray = None, 
                 mask: np.ndarray = None, use_temporal: bool = True) -> dict:
        """
        Compute multi-factor quality score.
        
        Args:
            img_rgb: RGB image (H, W, 3) or grayscale (H, W)
            seg_probs: Segmentation probabilities (C, H, W), optional
            mask: Valid region mask (H, W), optional
            use_temporal: Whether to apply temporal smoothing
        
        Returns:
            dict with quality score and individual factors
        """
        # Convert to grayscale if needed
        if img_rgb.ndim == 3:
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_rgb
        
        # Compute individual factors
        edge_sharpness = self._compute_edge_sharpness(img_gray, mask)
        contrast = self._compute_contrast(img_gray, mask)
        
        # Segmentation consistency (optional)
        if seg_probs is not None:
            seg_consistency = self._compute_segmentation_consistency(seg_probs, mask)
        else:
            seg_consistency = 0.5  # Neutral if no segmentation available
        
        # Weighted combination
        quality = (
            0.40 * edge_sharpness +
            0.30 * contrast +
            0.20 * seg_consistency +
            0.10 * 0.5  # Reserve 10% for temporal (placeholder)
        )
        
        # Temporal smoothing (optional)
        if use_temporal and self.prev_quality is not None:
            quality = (1 - self.temporal_weight) * quality + self.temporal_weight * self.prev_quality
        
        self.prev_quality = quality
        
        return {
            "quality": float(quality),
            "edge_sharpness": float(edge_sharpness),
            "contrast": float(contrast),
            "seg_consistency": float(seg_consistency),
        }
    
    def reset(self):
        """Reset temporal state"""
        self.prev_quality = None


def compute_quality_from_image(img_rgb: np.ndarray, seg_probs: np.ndarray = None, 
                                mask: np.ndarray = None) -> float:
    """
    Convenience function for single-frame quality computation.
    
    Args:
        img_rgb: RGB image (H, W, 3)
        seg_probs: Segmentation probabilities (C, H, W), optional
        mask: Valid region mask (H, W), optional
    
    Returns:
        quality score (0-1)
    """
    evaluator = MultiFactorQualityEvaluator()
    result = evaluator(img_rgb, seg_probs, mask, use_temporal=False)
    return result["quality"]
