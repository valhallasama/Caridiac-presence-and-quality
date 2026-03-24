"""
Multi-Task Training with Staged Strategy
Implements mathematical redesign with geometry-based presence and physics-based quality
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime

try:
    from model import MultiStructureGuidanceNet
    from dataset import CAMUSDataset, get_transforms
    from losses import MultiTaskLoss
    from ground_truth import GroundTruthConstructor, compute_dataset_statistics
except ImportError:
    from src.model import MultiStructureGuidanceNet
    from src.dataset import CAMUSDataset, get_transforms
    from src.losses import MultiTaskLoss
    from src.ground_truth import GroundTruthConstructor, compute_dataset_statistics


def dice_coeff_multi(pred, target, eps=1e-6):
    """Compute multi-class Dice coefficient"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(0, 2, 3))
    union = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean().item(), dice.detach().cpu().numpy()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs, gt_constructor):
    """Train for one epoch with staged strategy"""
    model.train()
    
    # Determine training stage
    if epoch < 50:
        stage = 'seg_only'
        stage_name = 'Stage 1: Segmentation Only'
    elif epoch < 150:
        stage = 'multi_task'
        stage_name = 'Stage 2: Multi-Task Learning'
    else:
        stage = 'full'
        stage_name = 'Stage 3: Full Refinement'
    
    running_losses = {
        'total': 0.0,
        'seg': 0.0,
        'presence': 0.0,
        'quality': 0.0,
        'consistency': 0.0,
        'anatomical': 0.0,
    }
    running_dice = 0.0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{total_epochs} [{stage_name}]')
    
    for batch_idx, batch_data in enumerate(pbar):
        # Unpack batch
        images = batch_data[0].to(device)
        masks = batch_data[1].to(device)
        
        # Compute ground truth for presence and quality
        batch_size = images.shape[0]
        presence_gt = []
        quality_gt = []
        
        for i in range(batch_size):
            # Convert to numpy for ground truth computation
            img_np = images[i].cpu().permute(1, 2, 0).numpy()
            mask_np = masks[i].cpu().numpy()
            
            # Compute GT
            gt = gt_constructor(img_np, mask_np)
            presence_gt.append(gt['presence'])
            quality_gt.append(gt['quality'])
        
        presence_gt = torch.tensor(presence_gt, dtype=torch.float32, device=device).unsqueeze(1)
        quality_gt = torch.tensor(quality_gt, dtype=torch.float32, device=device).unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Prepare targets
        targets = {
            'mask': masks,
            'presence': presence_gt,
            'quality': quality_gt,
        }
        
        # Compute loss
        losses = criterion(outputs, targets, stage=stage)
        loss = losses['total']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update running losses
        for key in running_losses.keys():
            if key in losses:
                running_losses[key] += losses[key].item()
        
        # Compute Dice for monitoring
        seg_probs = torch.sigmoid(outputs['seg'])
        dice, _ = dice_coeff_multi(seg_probs, masks)
        running_dice += dice
        
        # Update progress bar
        if batch_idx % 10 == 0:
            avg_loss = running_losses['total'] / (batch_idx + 1)
            avg_dice = running_dice / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'dice': f'{avg_dice:.4f}',
            })
    
    # Compute epoch averages
    num_batches = len(loader)
    epoch_losses = {k: v / num_batches for k, v in running_losses.items()}
    epoch_dice = running_dice / num_batches
    
    # Add actual prediction statistics (not just losses)
    epoch_stats = {
        'presence_pred_mean': outputs['presence'].mean().item() if 'presence' in outputs else 0.0,
        'presence_gt_mean': presence_gt.mean().item() if presence_gt is not None else 0.0,
        'quality_pred_mean': outputs['quality'].mean().item() if 'quality' in outputs else 0.0,
        'quality_gt_mean': quality_gt.mean().item() if quality_gt is not None else 0.0,
    }
    
    return epoch_losses, epoch_dice, epoch_stats


def validate(model, loader, criterion, device, gt_constructor):
    """Validate the model"""
    model.eval()
    
    running_losses = {
        'total': 0.0,
        'seg': 0.0,
        'presence': 0.0,
        'quality': 0.0,
    }
    running_dice = 0.0
    
    with torch.no_grad():
        for batch_data in tqdm(loader, desc='Validation'):
            # Unpack batch
            images = batch_data[0].to(device)
            masks = batch_data[1].to(device)
            
            # Compute ground truth
            batch_size = images.shape[0]
            presence_gt = []
            quality_gt = []
            
            for i in range(batch_size):
                img_np = images[i].cpu().permute(1, 2, 0).numpy()
                mask_np = masks[i].cpu().numpy()
                gt = gt_constructor(img_np, mask_np)
                presence_gt.append(gt['presence'])
                quality_gt.append(gt['quality'])
            
            presence_gt = torch.tensor(presence_gt, dtype=torch.float32, device=device).unsqueeze(1)
            quality_gt = torch.tensor(quality_gt, dtype=torch.float32, device=device).unsqueeze(1)
            
            # Forward pass
            outputs = model(images)
            
            # Prepare targets
            targets = {
                'mask': masks,
                'presence': presence_gt,
                'quality': quality_gt,
            }
            
            # Compute loss (always use full stage for validation)
            losses = criterion(outputs, targets, stage='multi_task')
            
            # Update running losses
            for key in running_losses.keys():
                if key in losses:
                    running_losses[key] += losses[key].item()
            
            # Compute Dice
            seg_probs = torch.sigmoid(outputs['seg'])
            dice, _ = dice_coeff_multi(seg_probs, masks)
            running_dice += dice
    
    # Compute averages
    num_batches = len(loader)
    val_losses = {k: v / num_batches for k, v in running_losses.items()}
    val_dice = running_dice / num_batches
    
    # Add actual prediction statistics
    val_stats = {
        'presence_pred_mean': outputs['presence'].mean().item() if 'presence' in outputs else 0.0,
        'presence_gt_mean': presence_gt.mean().item() if presence_gt is not None else 0.0,
        'quality_pred_mean': outputs['quality'].mean().item() if 'quality' in outputs else 0.0,
        'quality_gt_mean': quality_gt.mean().item() if quality_gt is not None else 0.0,
    }
    
    return val_losses, val_dice, val_stats


def main():
    parser = argparse.ArgumentParser(description='Multi-Task Training with Staged Strategy')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to CAMUS database_nifti')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/multitask_staged', help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, default=200, help='Total epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--synthetic_neg_prob', type=float, default=0.2, help='Synthetic negative probability')
    parser.add_argument('--synthetic_partial_prob', type=float, default=0.3, help='Synthetic partial probability')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--start_epoch', type=int, default=None, help='Override start epoch (useful for resuming from specific stage)')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    print('Creating datasets...')
    train_dataset = CAMUSDataset(
        data_dir=args.data_dir,
        transform=get_transforms('train', args.img_size),
        phase='train',
        img_size=args.img_size,
        synthetic_neg_prob=args.synthetic_neg_prob,
        synthetic_partial_prob=args.synthetic_partial_prob,
    )
    
    val_dataset = CAMUSDataset(
        data_dir=args.data_dir,
        transform=get_transforms('val', args.img_size),
        phase='val',
        img_size=args.img_size,
        synthetic_neg_prob=0.0,
        synthetic_partial_prob=0.0,
    )
    
    # Compute dataset statistics for ground truth construction
    print('Computing dataset statistics...')
    dataset_stats = compute_dataset_statistics(train_dataset)
    
    # Create ground truth constructor
    gt_constructor = GroundTruthConstructor(dataset_stats)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create model
    print('Creating model...')
    model = MultiStructureGuidanceNet(pretrained=True, num_structures=3, num_views=3)
    model = model.to(device)
    
    # Create multi-task loss
    criterion = MultiTaskLoss(
        seg_weight=1.0,
        presence_weight=0.5,
        quality_weight=0.5,
        consistency_weight=0.2,
        anatomical_weight=0.1,
    )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_dice = 0.0
    
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_dice = checkpoint.get('best_val_dice', 0.0)
        
        # Allow overriding start epoch (useful for resuming from specific stage)
        if args.start_epoch is not None:
            print(f'Overriding start epoch: {start_epoch} -> {args.start_epoch}')
            start_epoch = args.start_epoch
        
        print(f'Resumed from epoch {start_epoch}, best Dice: {best_val_dice:.4f}')
    
    # Training loop
    print(f'\nStarting training for {args.epochs} epochs...')
    print('=' * 80)
    print('STAGED TRAINING STRATEGY:')
    print('  Stage 1 (epochs 1-50):   Segmentation only (stabilize backbone)')
    print('  Stage 2 (epochs 51-150): Multi-task learning (seg + presence + quality)')
    print('  Stage 3 (epochs 151-200): Full refinement (+ consistency + anatomical)')
    print('=' * 80)
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_losses, train_dice, train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs, gt_constructor
        )
        
        # Validate
        val_losses, val_dice, val_stats = validate(
            model, val_loader, criterion, device, gt_constructor
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{args.epochs}:')
        print(f'  Train - Loss: {train_losses["total"]:.4f}, Dice: {train_dice:.4f}')
        print(f'  Val   - Loss: {val_losses["total"]:.4f}, Dice: {val_dice:.4f}')
        
        if epoch >= 50:  # Multi-task stage
            print(f'  Train - Presence Loss: {train_losses.get("presence", 0):.4f}, Quality Loss: {train_losses.get("quality", 0):.4f}')
            print(f'  Val   - Presence Loss: {val_losses.get("presence", 0):.4f}, Quality Loss: {val_losses.get("quality", 0):.4f}')
            print(f'  Train - Presence Pred/GT: {train_stats["presence_pred_mean"]:.3f}/{train_stats["presence_gt_mean"]:.3f}, Quality Pred/GT: {train_stats["quality_pred_mean"]:.3f}/{train_stats["quality_gt_mean"]:.3f}')
            print(f'  Val   - Presence Pred/GT: {val_stats["presence_pred_mean"]:.3f}/{val_stats["presence_gt_mean"]:.3f}, Quality Pred/GT: {val_stats["quality_pred_mean"]:.3f}/{val_stats["quality_gt_mean"]:.3f}')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses['total'],
            'val_loss': val_losses['total'],
            'train_dice': train_dice,
            'val_dice': val_dice,
            'best_val_dice': best_val_dice,
            'dataset_stats': dataset_stats,
        }
        
        # Save last checkpoint
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'last_checkpoint.pth'))
        
        # Save best checkpoint
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            checkpoint['best_val_dice'] = best_val_dice
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f'  *** New best model saved! Dice: {best_val_dice:.4f} ***')
        
        # Save stage checkpoints
        if epoch == 49:  # End of stage 1
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'stage1_final.pth'))
            print('  *** Stage 1 complete! Checkpoint saved. ***')
        elif epoch == 149:  # End of stage 2
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'stage2_final.pth'))
            print('  *** Stage 2 complete! Checkpoint saved. ***')
    
    print('\n' + '=' * 80)
    print('Training complete!')
    print(f'Best validation Dice: {best_val_dice:.4f}')
    print(f'Checkpoints saved to: {args.checkpoint_dir}')
    print('=' * 80)


if __name__ == '__main__':
    main()
