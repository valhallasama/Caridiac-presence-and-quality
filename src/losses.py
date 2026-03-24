"""
Multi-Task Training Loss for Presence and Quality
Implements mathematically correct loss functions that learn presence/quality
from segmentation-only supervision.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for segmentation"""
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        
    def forward(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        
        # Dice loss
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()
        
        # BCE loss
        bce = F.binary_cross_entropy(pred, target, reduction='mean')
        
        return dice_loss + bce


class PresenceLoss(nn.Module):
    """
    Presence loss with uncertainty-aware weighting.
    
    L_pres = C * SmoothL1(P_pred, P_gt)
    
    where C is segmentation confidence
    """
    def __init__(self, use_confidence_weighting=True):
        super(PresenceLoss, self).__init__()
        self.use_confidence = use_confidence_weighting
        
    def forward(self, pred, target, seg_probs=None):
        """
        Args:
            pred: (B, 1) predicted presence
            target: (B, 1) ground truth presence
            seg_probs: (B, C, H, W) segmentation probabilities (optional)
        """
        # Smooth L1 loss (more robust than MSE)
        loss = F.smooth_l1_loss(pred, target, reduction='none')
        
        # Confidence weighting (if segmentation is uncertain, reduce trust)
        if self.use_confidence and seg_probs is not None:
            # Compute segmentation confidence
            confidence = seg_probs.mean(dim=(1, 2, 3), keepdim=True)  # (B, 1)
            confidence = confidence.clamp(min=0.1, max=1.0)  # Avoid zero weight
            loss = loss * confidence
        
        return loss.mean()


class QualityLoss(nn.Module):
    """
    Quality loss (regression).
    
    L_qual = SmoothL1(Q_pred, Q_gt)
    """
    def __init__(self):
        super(QualityLoss, self).__init__()
        
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1) predicted quality
            target: (B, 1) ground truth quality
        """
        return F.smooth_l1_loss(pred, target)


class ConsistencyLoss(nn.Module):
    """
    Consistency loss: quality should be consistent with presence.
    
    L_cons = |Q_pred - P_pred * Q_pred|
    
    Enforces: if no heart (P→0), then quality must be low (Q→0)
    """
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        
    def forward(self, quality_pred, presence_pred):
        """
        Args:
            quality_pred: (B, 1) predicted quality
            presence_pred: (B, 1) predicted presence
        """
        # Quality should be scaled by presence
        expected_quality = presence_pred * quality_pred
        
        # Penalize deviation
        loss = torch.abs(quality_pred - expected_quality)
        
        return loss.mean()


class AnatomicalConstraintLoss(nn.Module):
    """
    Anatomical constraint loss: enforce structural relationships.
    
    Constraints:
    - LV should be inside Myo (roughly)
    - Relative size ratios should be reasonable
    """
    def __init__(self):
        super(AnatomicalConstraintLoss, self).__init__()
        
    def forward(self, seg_probs):
        """
        Args:
            seg_probs: (B, 3, H, W) with [LV, Myo, LA]
        """
        lv = seg_probs[:, 0]    # (B, H, W)
        myo = seg_probs[:, 1]   # (B, H, W)
        la = seg_probs[:, 2]    # (B, H, W)
        
        # Compute areas
        lv_area = lv.sum(dim=(1, 2))    # (B,)
        myo_area = myo.sum(dim=(1, 2))  # (B,)
        la_area = la.sum(dim=(1, 2))    # (B,)
        
        # Only apply constraints when structures are present (avoid division by near-zero)
        # This is critical for synthetic negatives/partials
        min_area_threshold = 100.0  # Minimum pixels to consider structure present
        
        # Constraint 1: LV should be smaller than Myo (when both present)
        # Ratio should be in reasonable range (0.3 - 0.8)
        valid_lv_myo = (lv_area > min_area_threshold) & (myo_area > min_area_threshold)
        if valid_lv_myo.any():
            ratio = lv_area[valid_lv_myo] / (myo_area[valid_lv_myo] + 1e-6)
            ratio = torch.clamp(ratio, 0.0, 2.0)  # Clip to prevent explosion
            ratio_loss = F.relu(ratio - 0.8) + F.relu(0.3 - ratio)
            ratio_loss = ratio_loss.mean()
        else:
            ratio_loss = torch.tensor(0.0, device=seg_probs.device)
        
        # Constraint 2: LA should be reasonable relative to LV (when both present)
        # Ratio should be in range (0.2 - 1.5)
        valid_la_lv = (la_area > min_area_threshold) & (lv_area > min_area_threshold)
        if valid_la_lv.any():
            la_lv_ratio = la_area[valid_la_lv] / (lv_area[valid_la_lv] + 1e-6)
            la_lv_ratio = torch.clamp(la_lv_ratio, 0.0, 3.0)  # Clip to prevent explosion
            la_loss = F.relu(la_lv_ratio - 1.5) + F.relu(0.2 - la_lv_ratio)
            la_loss = la_loss.mean()
        else:
            la_loss = torch.tensor(0.0, device=seg_probs.device)
        
        return ratio_loss + la_loss


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for segmentation + presence + quality.
    
    L = λ1*L_seg + λ2*L_pres + λ3*L_qual + λ4*L_cons + λ5*L_anat
    """
    def __init__(self, 
                 seg_weight=1.0,
                 presence_weight=0.5,
                 quality_weight=0.5,
                 consistency_weight=0.2,
                 anatomical_weight=0.1,
                 use_confidence_weighting=True):
        super(MultiTaskLoss, self).__init__()
        
        self.seg_loss = DiceBCELoss()
        self.presence_loss = PresenceLoss(use_confidence_weighting)
        self.quality_loss = QualityLoss()
        self.consistency_loss = ConsistencyLoss()
        self.anatomical_loss = AnatomicalConstraintLoss()
        
        self.seg_weight = seg_weight
        self.presence_weight = presence_weight
        self.quality_weight = quality_weight
        self.consistency_weight = consistency_weight
        self.anatomical_weight = anatomical_weight
        
    def forward(self, outputs, targets, stage='full'):
        """
        Args:
            outputs: dict with keys:
                - 'seg': (B, 3, H, W) segmentation logits
                - 'presence': (B, 1) presence prediction
                - 'quality': (B, 1) quality prediction
            targets: dict with keys:
                - 'mask': (B, 3, H, W) segmentation ground truth
                - 'presence': (B, 1) presence ground truth
                - 'quality': (B, 1) quality ground truth
            stage: 'seg_only', 'multi_task', or 'full'
        
        Returns:
            dict with total loss and individual components
        """
        losses = {}
        
        # Segmentation loss (always computed)
        seg_logits = outputs['seg']
        seg_target = targets['mask']
        losses['seg'] = self.seg_loss(seg_logits, seg_target)
        
        # Compute segmentation probabilities for confidence weighting
        seg_probs = torch.sigmoid(seg_logits)
        
        if stage in ['multi_task', 'full']:
            # Presence loss
            presence_pred = outputs['presence']
            presence_target = targets['presence']
            losses['presence'] = self.presence_loss(
                presence_pred, presence_target, seg_probs
            )
            
            # Quality loss
            quality_pred = outputs['quality']
            quality_target = targets['quality']
            losses['quality'] = self.quality_loss(quality_pred, quality_target)
        
        if stage == 'full':
            # Consistency loss
            losses['consistency'] = self.consistency_loss(
                outputs['quality'], outputs['presence']
            )
            
            # Anatomical constraint loss
            losses['anatomical'] = self.anatomical_loss(seg_probs)
        
        # Weighted combination
        total_loss = self.seg_weight * losses['seg']
        
        if 'presence' in losses:
            total_loss += self.presence_weight * losses['presence']
        if 'quality' in losses:
            total_loss += self.quality_weight * losses['quality']
        if 'consistency' in losses:
            total_loss += self.consistency_weight * losses['consistency']
        if 'anatomical' in losses:
            total_loss += self.anatomical_weight * losses['anatomical']
        
        losses['total'] = total_loss
        
        return losses


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for video sequences.
    
    L_temporal = |P_t - P_{t-1}| + |Q_t - Q_{t-1}|
    
    Enforces smooth transitions over time.
    """
    def __init__(self, alpha=0.3):
        super(TemporalConsistencyLoss, self).__init__()
        self.alpha = alpha  # Weight for temporal smoothing
        
    def forward(self, current, previous):
        """
        Args:
            current: dict with 'presence' and 'quality' (B, 1)
            previous: dict with 'presence' and 'quality' (B, 1)
        """
        if previous is None:
            return torch.tensor(0.0, device=current['presence'].device)
        
        presence_diff = torch.abs(current['presence'] - previous['presence'])
        quality_diff = torch.abs(current['quality'] - previous['quality'])
        
        return self.alpha * (presence_diff.mean() + quality_diff.mean())


# Convenience function for creating loss
def create_multitask_loss(config=None):
    """
    Create multi-task loss with default or custom configuration.
    
    Args:
        config: dict with loss weights (optional)
    
    Returns:
        MultiTaskLoss instance
    """
    if config is None:
        config = {
            'seg_weight': 1.0,
            'presence_weight': 0.5,
            'quality_weight': 0.5,
            'consistency_weight': 0.2,
            'anatomical_weight': 0.1,
        }
    
    return MultiTaskLoss(**config)
