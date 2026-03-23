"""
Ground Truth Construction for Presence and Quality
Derives presence and quality labels from CAMUS segmentation masks + image statistics
Based on mathematical redesign to decouple from segmentation dependency
"""
import numpy as np
import torch
import cv2
from scipy.ndimage import laplace


class GroundTruthConstructor:
    """
    Constructs presence and quality ground truth from:
    - Segmentation masks (structure information)
    - Image statistics (physics-based quality)
    
    This allows training presence/quality heads with only segmentation labels.
    """
    
    def __init__(self, dataset_stats=None):
        """
        Args:
            dataset_stats: dict with reference areas for normalization
                {'lv_ref': float, 'myo_ref': float, 'la_ref': float}
        """
        # Default reference areas (will be computed from dataset if not provided)
        self.stats = dataset_stats or {
            'lv_ref': 9830.0,   # ~15% of 256x256
            'myo_ref': 16384.0,  # ~25% of 256x256
            'la_ref': 6554.0,    # ~10% of 256x256
        }
        
        # Structure weights for presence calculation
        self.structure_weights = {
            'lv': 0.5,
            'myo': 0.3,
            'la': 0.2,
        }
        
        # Quality component weights
        self.quality_weights = {
            'sharpness': 0.4,
            'contrast': 0.3,
            'structure': 0.3,
        }
    
    def compute_presence_gt(self, mask: np.ndarray) -> float:
        """
        Geometry-based presence (independent of intensity).
        
        P_gt = Σ w_i * (A_i / A_i_ref)
        
        Args:
            mask: (H, W, 3) or (3, H, W) with [LV, Myo, LA] channels
        
        Returns:
            presence score in [0, 1]
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Ensure (3, H, W) format
        if mask.ndim == 3 and mask.shape[-1] == 3:
            mask = mask.transpose(2, 0, 1)
        
        # Compute areas for each structure
        areas = {
            'lv': float(mask[0].sum()),
            'myo': float(mask[1].sum()),
            'la': float(mask[2].sum()),
        }
        
        # Normalize by reference areas
        normalized_areas = {
            k: areas[k] / self.stats[f'{k}_ref']
            for k in ['lv', 'myo', 'la']
        }
        
        # Weighted sum
        presence = sum(
            self.structure_weights[k] * normalized_areas[k]
            for k in ['lv', 'myo', 'la']
        )
        
        # Clip to [0, 1]
        presence = np.clip(presence, 0.0, 1.0)
        
        return float(presence)
    
    def compute_sharpness(self, img: np.ndarray, mask: np.ndarray = None) -> float:
        """
        Edge sharpness using Laplacian variance.
        
        Q_sharp = Var(∇²I)
        
        Args:
            img: (H, W) or (H, W, 3) image
            mask: (H, W) valid region mask (optional)
        
        Returns:
            sharpness score in [0, 1]
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Convert to grayscale if needed
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Compute Laplacian
        laplacian = laplace(img.astype(np.float32))
        
        # Apply mask if provided
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            # Combine all structure channels
            if mask.ndim == 3:
                mask = (mask.sum(axis=0) > 0).astype(np.float32)
            laplacian = laplacian * mask
            valid_pixels = mask.sum()
            if valid_pixels < 100:
                return 0.0
        
        # Variance of Laplacian
        variance = float(laplacian.var())
        
        # Normalize to [0, 1] (empirical range: 0-100 for typical ultrasound)
        sharpness = np.clip(variance / 100.0, 0.0, 1.0)
        
        return sharpness
    
    def compute_contrast(self, img: np.ndarray, mask: np.ndarray = None) -> float:
        """
        Local contrast using coefficient of variation.
        
        Q_contrast = σ_I / (μ_I + ε)
        
        Args:
            img: (H, W) or (H, W, 3) image
            mask: (H, W) valid region mask (optional)
        
        Returns:
            contrast score in [0, 1]
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Convert to grayscale if needed
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply mask if provided
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            # Combine all structure channels
            if mask.ndim == 3:
                mask = (mask.sum(axis=0) > 0).astype(np.float32)
            valid_pixels = img[mask > 0]
            if len(valid_pixels) < 100:
                return 0.0
        else:
            valid_pixels = img.flatten()
        
        # Coefficient of variation
        mean = float(valid_pixels.mean())
        std = float(valid_pixels.std())
        
        if mean < 1e-6:
            return 0.0
        
        contrast = std / mean
        
        # Normalize to [0, 1] (empirical range: 0-1.5 for typical ultrasound)
        contrast = np.clip(contrast / 1.5, 0.0, 1.0)
        
        return contrast
    
    def compute_quality_gt(self, img: np.ndarray, mask: np.ndarray, 
                          presence_gt: float = None) -> float:
        """
        Physics-based quality (independent of segmentation confidence).
        
        Q_gt = α*Q_sharp + β*Q_contrast + γ*Q_struct
        
        Args:
            img: (H, W) or (H, W, 3) image
            mask: (H, W, 3) or (3, H, W) segmentation mask
            presence_gt: pre-computed presence (optional)
        
        Returns:
            quality score in [0, 1]
        """
        # Compute presence if not provided
        if presence_gt is None:
            presence_gt = self.compute_presence_gt(mask)
        
        # Compute image-based metrics
        sharpness = self.compute_sharpness(img, mask)
        contrast = self.compute_contrast(img, mask)
        
        # Structure completeness = presence
        structure = presence_gt
        
        # Weighted combination
        quality = (
            self.quality_weights['sharpness'] * sharpness +
            self.quality_weights['contrast'] * contrast +
            self.quality_weights['structure'] * structure
        )
        
        return float(quality)
    
    def __call__(self, img: np.ndarray, mask: np.ndarray) -> dict:
        """
        Compute both presence and quality ground truth.
        
        Args:
            img: (H, W) or (H, W, 3) image
            mask: (H, W, 3) or (3, H, W) segmentation mask
        
        Returns:
            dict with 'presence', 'quality', and intermediate metrics
        """
        presence = self.compute_presence_gt(mask)
        quality = self.compute_quality_gt(img, mask, presence)
        
        return {
            'presence': presence,
            'quality': quality,
            'sharpness': self.compute_sharpness(img, mask),
            'contrast': self.compute_contrast(img, mask),
        }


def compute_dataset_statistics(dataset):
    """
    Compute reference areas from CAMUS dataset for normalization.
    
    Args:
        dataset: CAMUSDataset instance
    
    Returns:
        dict with reference areas
    """
    lv_areas = []
    myo_areas = []
    la_areas = []
    
    print("Computing dataset statistics...")
    for i in range(min(len(dataset), 1000)):  # Sample up to 1000 images
        try:
            sample = dataset[i]
            if isinstance(sample, tuple):
                mask = sample[1]  # Assuming (img, mask, ...)
            else:
                continue
            
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # Ensure (3, H, W) format
            if mask.ndim == 3 and mask.shape[-1] == 3:
                mask = mask.transpose(2, 0, 1)
            
            lv_areas.append(float(mask[0].sum()))
            myo_areas.append(float(mask[1].sum()))
            la_areas.append(float(mask[2].sum()))
        except Exception:
            continue
    
    stats = {
        'lv_ref': float(np.mean(lv_areas)) if lv_areas else 9830.0,
        'myo_ref': float(np.mean(myo_areas)) if myo_areas else 16384.0,
        'la_ref': float(np.mean(la_areas)) if la_areas else 6554.0,
    }
    
    print(f"Dataset statistics computed:")
    print(f"  LV reference area: {stats['lv_ref']:.1f}")
    print(f"  Myo reference area: {stats['myo_ref']:.1f}")
    print(f"  LA reference area: {stats['la_ref']:.1f}")
    
    return stats
