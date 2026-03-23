import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import MultiStructureGuidanceNet
from src.dataset import CAMUSDataset, get_transforms
from src.train import train_one_epoch, validate

def dry_run():
    # Configuration
    DATA_DIR = "data/CAMUS_public/database_nifti"
    BATCH_SIZE = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found!")
        return

    # Datasets
    # Use a tiny subset
    train_ds = CAMUSDataset(DATA_DIR, transform=get_transforms('train'), phase='train')
    val_ds = CAMUSDataset(DATA_DIR, transform=get_transforms('val'), phase='val')
    
    # Mocking smaller dataset for speed
    train_ds.samples = train_ds.samples[:10]
    val_ds.samples = val_ds.samples[:4]
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    b = next(iter(train_loader))
    print(f"Batch items: {len(b)}")
    print(f"  img: {b[0].shape}  mask: {b[1].shape}  quality: {b[3].shape}  view: {b[4].shape}  presence: {b[5].shape}")
    
    # Model
    model = MultiStructureGuidanceNet(num_structures=3, num_views=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting dry run training (1 epoch)...")
    train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
    print(f"Train Loss: {train_loss:.4f}")
    
    print("Starting dry run validation...")
    val_metrics = validate(model, val_loader, DEVICE)
    print(f"Val Metrics: {val_metrics}")
    
    print("Dry run completed successfully!")

if __name__ == "__main__":
    dry_run()
