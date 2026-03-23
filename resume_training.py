"""
Resume Training from Checkpoint
Loads best model and continues training
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

try:
    from model import MultiStructureGuidanceNet
    from dataset import CAMUSDataset, get_transforms
    from train import train_one_epoch, validate, compute_quality_loss
except ImportError:
    from src.model import MultiStructureGuidanceNet
    from src.dataset import CAMUSDataset, get_transforms
    from src.train import train_one_epoch, validate, compute_quality_loss


def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to checkpoint (e.g., checkpoints/best_model_multi.pth)')
    parser.add_argument('--data-dir', type=str, default="data/CAMUS_public/database_nifti")
    parser.add_argument('--epochs', type=int, default=50, help='Additional epochs to train')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-half-sequence', action='store_true')
    parser.add_argument('--ckpt-tag', type=str, default='resumed')
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

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                if f.endswith('.pth'):
                    print(f"  checkpoints/{f}")
        return
    
    print(f"\n{'='*80}")
    print(f"RESUMING TRAINING FROM CHECKPOINT")
    print(f"{'='*80}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Additional epochs: {args.epochs}")
    print(f"Device: {DEVICE}")
    
    # Setup
    quality_loss_type = args.quality_loss
    if quality_loss_type is None:
        quality_loss_type = 'mse' if str(args.quality_source).lower() == 'camus_label' else 'smooth_l1'

    view_loss_weight = args.view_loss_weight
    if view_loss_weight is None:
        view_loss_weight = 0.0 if args.view_filter is not None else 0.25
    
    # Datasets
    print("\nLoading datasets...")
    train_ds = CAMUSDataset(
        args.data_dir,
        transform=get_transforms('train'),
        phase='train',
        use_half_sequence=not args.no_half_sequence,
        quality_source=args.quality_source,
        view_filter=args.view_filter,
        synthetic_neg_prob=args.synthetic_neg_prob,
        synthetic_partial_prob=args.synthetic_partial_prob,
        synthetic_inpaint_radius=args.synthetic_inpaint_radius,
        synthetic_neg_quality=args.synthetic_neg_quality,
        synthetic_partial_quality=args.synthetic_partial_quality,
    )
    val_ds = CAMUSDataset(
        args.data_dir,
        transform=get_transforms('val'),
        phase='val',
        use_half_sequence=not args.no_half_sequence,
        quality_source=args.quality_source,
        view_filter=args.view_filter,
        synthetic_neg_prob=0.0,
        synthetic_partial_prob=0.0,
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    
    # Model
    print("\nInitializing model...")
    model = MultiStructureGuidanceNet(num_structures=3, num_views=2).to(DEVICE)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    
    # Check if it's a full checkpoint or just model weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("✓ Full checkpoint detected (includes optimizer state)")
        model_state = checkpoint['model_state_dict']
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_dice_loaded = checkpoint.get('best_val_dice', 0.0)
        print(f"  Previous epoch: {checkpoint.get('epoch', 0)}")
        print(f"  Previous best Dice: {best_val_dice_loaded:.4f}")
    else:
        print("⚠ Old checkpoint format (model weights only)")
        model_state = checkpoint
        start_epoch = 0
        best_val_dice_loaded = 0.0
    
    # Try loading model with strict=False to handle architecture changes
    try:
        model.load_state_dict(model_state, strict=True)
        print("✓ Model weights loaded successfully (exact match)!")
    except RuntimeError as e:
        print(f"⚠ Architecture mismatch detected: {str(e)[:100]}...")
        print("  Attempting to load with strict=False (partial loading)...")
        
        # Load what matches, ignore mismatches
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        
        print(f"✓ Model weights loaded with partial matching!")
        if missing_keys:
            print(f"  Missing keys (will use random init): {len(missing_keys)}")
            for key in missing_keys[:5]:
                print(f"    - {key}")
            if len(missing_keys) > 5:
                print(f"    ... and {len(missing_keys)-5} more")
        
        if unexpected_keys:
            print(f"  Unexpected keys (ignored): {len(unexpected_keys)}")
            for key in unexpected_keys[:5]:
                print(f"    - {key}")
            if len(unexpected_keys) > 5:
                print(f"    ... and {len(unexpected_keys)-5} more")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load optimizer state if available
    if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state restored")
        except Exception as e:
            print(f"⚠ Could not restore optimizer state: {e}")
            print("  Using fresh optimizer")
    
    # Evaluate loaded model first
    print("\nEvaluating loaded model...")
    initial_metrics = validate(
        model,
        val_loader,
        DEVICE,
        quality_loss_type=quality_loss_type,
        view_loss_weight=view_loss_weight,
        presence_loss_weight=float(args.presence_loss_weight),
    )
    
    print(f"Initial Performance:")
    print(f"  Loss: {initial_metrics['loss']:.4f}")
    print(f"  Dice: {initial_metrics['dice']:.4f}")
    print(f"  Centroid Error: {initial_metrics['centroid_error']:.1f}px")
    print(f"  Quality MAE: {initial_metrics['quality_mae']:.4f}")
    print(f"  View Acc: {initial_metrics['view_acc']:.4f}")
    print(f"  Presence Acc: {initial_metrics['presence_acc']:.4f}")
    
    # Setup for continued training
    os.makedirs("checkpoints", exist_ok=True)
    best_val_dice = initial_metrics['dice']  # Start from current performance
    
    best_path = os.path.join("checkpoints", f"best_model_{args.ckpt_tag}.pth")
    last_path = os.path.join("checkpoints", f"last_model_{args.ckpt_tag}.pth")
    
    print(f"\n{'='*80}")
    print(f"STARTING CONTINUED TRAINING")
    print(f"{'='*80}")
    print(f"Best Dice to beat: {best_val_dice:.4f}")
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

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
        
        # Save best model
        if val_metrics['dice'] > best_val_dice:
            improvement = val_metrics['dice'] - best_val_dice
            best_val_dice = val_metrics['dice']
            torch.save(model.state_dict(), best_path)
            print(f"✓ New best model! Dice improved by {improvement:.4f}")
            
        # Save last
        torch.save(model.state_dict(), last_path)
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Initial Dice: {initial_metrics['dice']:.4f}")
    print(f"Final Best Dice: {best_val_dice:.4f}")
    print(f"Improvement: {best_val_dice - initial_metrics['dice']:.4f}")
    print(f"\nModels saved:")
    print(f"  Best: {best_path}")
    print(f"  Last: {last_path}")


if __name__ == "__main__":
    main()
