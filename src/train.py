import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

try:
    from model import MultiStructureGuidanceNet
    from dataset import CAMUSDataset, get_transforms
except ImportError:
    from src.model import MultiStructureGuidanceNet
    from src.dataset import CAMUSDataset, get_transforms

# --- Metrics ---
def dice_coeff(pred, target, eps=1e-6):
    # pred: (B, 1, H, W) sigmoid output
    # target: (B, 1, H, W) binary
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean().item()


def dice_coeff_multi(pred, target, eps=1e-6):
    # pred/target: (B, C, H, W)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(0, 2, 3))
    union = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean().item(), dice.detach().cpu().numpy()

def calculate_centroid_error(pred_mask, gt_mask):
    # pred_mask: (H, W)
    # gt_mask: (H, W)
    # Returns error in pixels, or None if empty
    
    def get_centroid(mask):
        indices = torch.nonzero(mask)
        if len(indices) == 0:
            return None
        return indices.float().mean(dim=0) # (y, x)

    pred_c = get_centroid(pred_mask > 0.5)
    gt_c = get_centroid(gt_mask > 0.5)
    
    if pred_c is None or gt_c is None:
        return None
        
    dist = torch.norm(pred_c - gt_c).item()
    return dist

# --- Loss Functions ---
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs: sigmoid output
        # targets: binary
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # BCE
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # Dice
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return bce + dice_loss

import torch.nn.functional as F


def compute_quality_loss(quality_logits, qualities, loss_type: str):
    quality_probs = torch.sigmoid(quality_logits)
    lt = str(loss_type).lower().strip()
    if lt == "mse":
        return F.mse_loss(quality_probs, qualities)
    return F.smooth_l1_loss(quality_probs, qualities)

# --- Training Engine ---
def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    quality_loss_type: str = "smooth_l1",
    view_loss_weight: float = 0.25,
    presence_loss_weight: float = 0.3,
):
    model.train()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_quality_loss = 0.0
    running_view_loss = 0.0
    running_presence_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    
    for imgs, masks, struct_presences, qualities, views, presences, camus_qualities in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)
        qualities = qualities.to(device)
        camus_qualities = camus_qualities.to(device)
        views = views.to(device)
        presences = presences.to(device)
        
        optimizer.zero_grad()
        
        out = model(imgs)
        seg_logits = out["seg"]
        presence_logits = out["presence"]
        quality_logits = out["quality"]
        camus_quality_logits = out["camus_quality"]
        view_pred = out["view"]
        aux_logits = out["aux"]
        
        # Losses
        # 1. Main Segmentation: Dice + BCE (multi-channel)
        bce_seg = F.binary_cross_entropy_with_logits(seg_logits, masks)
        seg_probs = torch.sigmoid(seg_logits)
        intersection = (seg_probs * masks).sum(dim=(0, 2, 3))
        union = seg_probs.sum(dim=(0, 2, 3)) + masks.sum(dim=(0, 2, 3))
        dice_loss = 1 - (2. * intersection + 1e-6) / (union + 1e-6)
        loss_seg = bce_seg + dice_loss.mean()
        
        # 2. Aux Segmentation: BCE (Deep Supervision)
        masks_ds = F.interpolate(masks, size=aux_logits.shape[-2:], mode='nearest')
        bce_aux = F.binary_cross_entropy_with_logits(aux_logits, masks_ds)
        aux_probs = torch.sigmoid(aux_logits)
        intersection_aux = (aux_probs * masks_ds).sum(dim=(0, 2, 3))
        union_aux = aux_probs.sum(dim=(0, 2, 3)) + masks_ds.sum(dim=(0, 2, 3))
        dice_aux = 1 - (2. * intersection_aux + 1e-6) / (union_aux + 1e-6)
        loss_aux = bce_aux + dice_aux.mean()

        # 3. Quality: regression (derived from segmentation - continuous)
        loss_quality = compute_quality_loss(quality_logits, qualities, quality_loss_type)
        
        # 3a. CAMUS Quality: regression (expert labels)
        loss_camus_quality = compute_quality_loss(camus_quality_logits, camus_qualities, 'mse')

        # 3b. Presence: BCE (learned)
        loss_presence = F.binary_cross_entropy_with_logits(presence_logits, presences)
        
        # 4. View: CrossEntropy
        if float(view_loss_weight) > 0.0:
            loss_view = F.cross_entropy(view_pred, views)
        else:
            loss_view = torch.zeros((), dtype=loss_quality.dtype, device=loss_quality.device)
        
        # Total loss
        loss = (
            2.0 * loss_seg
            + 0.5 * loss_aux
            + 0.5 * loss_quality
            + 0.3 * loss_camus_quality
            + float(presence_loss_weight) * loss_presence
            + float(view_loss_weight) * loss_view
        )
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_seg_loss += loss_seg.item()
        running_quality_loss += loss_quality.item()
        running_view_loss += loss_view.item()
        running_presence_loss += loss_presence.item()
        
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(loader)

def validate(
    model,
    loader,
    device,
    quality_loss_type: str = "smooth_l1",
    view_loss_weight: float = 0.25,
    presence_loss_weight: float = 0.3,
):
    model.eval()
    running_loss = 0.0
    
    # Metrics
    dice_scores = []
    dice_per_class = []
    quality_targets = []
    quality_preds = []
    view_targets = []
    view_preds = []
    presence_targets = []
    presence_preds = []
    centroid_errors = []
    
    with torch.no_grad():
        for imgs, masks, struct_presences, qualities, views, presences, camus_qualities in tqdm(loader, desc="Validation"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            qualities = qualities.to(device)
            camus_qualities = camus_qualities.to(device)
            views = views.to(device)
            presences = presences.to(device)
            
            out = model(imgs)
            seg_logits = out["seg"]
            presence_logits = out["presence"]
            quality_logits = out["quality"]
            camus_quality_logits = out["camus_quality"]
            view_pred = out["view"]
            
            # Loss (monitor only)
            bce_seg = F.binary_cross_entropy_with_logits(seg_logits, masks)
            seg_probs = torch.sigmoid(seg_logits)
            intersection = (seg_probs * masks).sum(dim=(0, 2, 3))
            union = seg_probs.sum(dim=(0, 2, 3)) + masks.sum(dim=(0, 2, 3))
            dice_loss = 1 - (2. * intersection + 1e-6) / (union + 1e-6)
            loss_seg = bce_seg + dice_loss.mean()
            loss_quality = compute_quality_loss(quality_logits, qualities, quality_loss_type)
            loss_camus_quality = compute_quality_loss(camus_quality_logits, camus_qualities, 'mse')
            loss_presence = F.binary_cross_entropy_with_logits(presence_logits, presences)
            if float(view_loss_weight) > 0.0:
                loss_view = F.cross_entropy(view_pred, views)
            else:
                loss_view = torch.zeros((), dtype=loss_quality.dtype, device=loss_quality.device)
            loss = (
                1.0 * loss_seg
                + 0.5 * loss_quality
                + 0.3 * loss_camus_quality
                + float(presence_loss_weight) * loss_presence
                + float(view_loss_weight) * loss_view
            )
            running_loss += loss.item()
            
            # Segmentation Metric: Dice
            batch_dice_mean, batch_dice_per_class = dice_coeff_multi(seg_probs, masks)
            dice_scores.append(batch_dice_mean)
            dice_per_class.append(batch_dice_per_class)
            
            # Segmentation Metric: Centroid Error (pixel)
            for i in range(masks.size(0)):
                err = calculate_centroid_error(seg_probs[i, 0], masks[i, 0])
                if err is not None:
                    centroid_errors.append(err)
            
            # Quality Metric
            quality_targets.append(qualities.cpu().numpy())
            quality_preds.append(torch.sigmoid(quality_logits).cpu().numpy())
            
            # View Metric
            view_targets.extend(views.cpu().numpy())
            view_preds.extend(torch.argmax(view_pred, dim=1).cpu().numpy())

            # Presence Metric
            presence_targets.append(presences.cpu().numpy())
            presence_preds.append(torch.sigmoid(presence_logits).cpu().numpy())
            
    # Aggregate
    avg_loss = running_loss / len(loader)
    avg_dice = np.mean(dice_scores)
    avg_centroid_err = np.mean(centroid_errors) if centroid_errors else 0.0

    dice_pc = np.mean(np.stack(dice_per_class, axis=0), axis=0) if dice_per_class else None

    quality_targets = np.concatenate(quality_targets, axis=0) if quality_targets else None
    quality_preds = np.concatenate(quality_preds, axis=0) if quality_preds else None
    quality_mae = float(np.mean(np.abs(quality_targets - quality_preds))) if quality_targets is not None else 0.0

    presence_targets = np.concatenate(presence_targets, axis=0) if presence_targets else None
    presence_preds = np.concatenate(presence_preds, axis=0) if presence_preds else None
    if presence_targets is not None and presence_preds is not None:
        presence_acc = float(np.mean((presence_preds >= 0.5) == (presence_targets >= 0.5)))
    else:
        presence_acc = 0.0
        
    acc_view = accuracy_score(view_targets, view_preds)
    
    return {
        'loss': avg_loss,
        'dice': avg_dice,
        'dice_per_class': dice_pc,
        'centroid_error': avg_centroid_err,
        'quality_mae': quality_mae,
        'view_acc': acc_view,
        'presence_acc': presence_acc,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="data/CAMUS_public/database_nifti")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-half-sequence', action='store_true')
    parser.add_argument('--ckpt-tag', type=str, default='multi')
    parser.add_argument('--quality-source', type=str, default='derived', choices=['derived', 'camus_label'])
    parser.add_argument('--quality-loss', type=str, default=None, choices=['smooth_l1', 'mse'])
    parser.add_argument('--view-filter', type=str, default=None, choices=['2CH', '4CH'])
    parser.add_argument('--view-loss-weight', type=float, default=None)
    parser.add_argument('--presence-loss-weight', type=float, default=0.3)
    parser.add_argument('--synthetic-neg-prob', type=float, default=0.0)
    parser.add_argument('--synthetic-partial-prob', type=float, default=0.0)
    parser.add_argument('--synthetic-inpaint-radius', type=int, default=5)
    parser.add_argument('--synthetic-neg-quality', type=float, default=0.0)
    parser.add_argument('--synthetic-partial-quality', type=float, default=0.3)
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    NUM_WORKERS = args.num_workers
    USE_HALF_SEQUENCE = not args.no_half_sequence
    CKPT_TAG = args.ckpt_tag
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    quality_loss_type = args.quality_loss
    if quality_loss_type is None:
        quality_loss_type = 'mse' if str(args.quality_source).lower() == 'camus_label' else 'smooth_l1'

    view_loss_weight = args.view_loss_weight
    if view_loss_weight is None:
        view_loss_weight = 0.0 if args.view_filter is not None else 0.25
    
    print(f"Using device: {DEVICE}")
    
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found!")
        return

    # Datasets
    train_ds = CAMUSDataset(
        DATA_DIR,
        transform=get_transforms('train'),
        phase='train',
        use_half_sequence=USE_HALF_SEQUENCE,
        quality_source=args.quality_source,
        view_filter=args.view_filter,
        synthetic_neg_prob=args.synthetic_neg_prob,
        synthetic_partial_prob=args.synthetic_partial_prob,
        synthetic_inpaint_radius=args.synthetic_inpaint_radius,
        synthetic_neg_quality=args.synthetic_neg_quality,
        synthetic_partial_quality=args.synthetic_partial_quality,
    )
    val_ds = CAMUSDataset(
        DATA_DIR,
        transform=get_transforms('val'),
        phase='val',
        use_half_sequence=USE_HALF_SEQUENCE,
        quality_source=args.quality_source,
        view_filter=args.view_filter,
        synthetic_neg_prob=0.0,
        synthetic_partial_prob=0.0,
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Model
    model = MultiStructureGuidanceNet(num_structures=3, num_views=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    best_val_dice = 0.0

    best_path = os.path.join("checkpoints", f"best_model_{CKPT_TAG}.pth")
    last_path = os.path.join("checkpoints", f"last_model_{CKPT_TAG}.pth")
    checkpoint_dir = os.path.join("checkpoints", f"training_{CKPT_TAG}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            DEVICE,
            quality_loss_type=quality_loss_type,
            view_loss_weight=view_loss_weight,
            presence_loss_weight=float(args.presence_loss_weight),
        )
        val_metrics = validate(
            model,
            val_loader,
            DEVICE,
            quality_loss_type=quality_loss_type,
            view_loss_weight=view_loss_weight,
            presence_loss_weight=float(args.presence_loss_weight),
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Metrics: Loss={val_metrics['loss']:.4f}, Dice={val_metrics['dice']:.4f}, "
              f"CentroidErr={val_metrics['centroid_error']:.1f}px, "
              f"QualityMAE={val_metrics['quality_mae']:.4f}, "
              f"ViewAcc={val_metrics['view_acc']:.4f}, "
              f"PresenceAcc={val_metrics['presence_acc']:.4f}")
        
        # Save best model (with full checkpoint)
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_dice': best_val_dice,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, best_path)
            print(f"Saved best model! (Dice: {best_val_dice:.4f})")
            
        # Save last model (with full checkpoint)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_dice': best_val_dice,
            'val_metrics': val_metrics
        }
        torch.save(checkpoint, last_path)
        
        # Save periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            periodic_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, periodic_path)
            print(f"Saved checkpoint at epoch {epoch+1}")

if __name__ == "__main__":
    main()
