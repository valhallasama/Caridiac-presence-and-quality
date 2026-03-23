from src.dataset import CAMUSDataset, get_transforms
import torch
import os

def verify_data():
    data_dir = "data/CAMUS_public/database_nifti"
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return

    print("Initializing dataset...")
    try:
        ds = CAMUSDataset(data_dir, transform=get_transforms('train'), phase='train')
        print(f"Dataset size: {len(ds)}")
        
        if len(ds) > 0:
            img, mask, struct_pres, quality, view, presence = ds[0]
            print(f"Sample 0:")
            print(f"  Image shape: {img.shape} (Type: {img.dtype})")
            print(f"  Mask shape: {mask.shape} (Type: {mask.dtype})")
            print(f"  Structure presence: {struct_pres} (Type: {struct_pres.dtype})")
            print(f"  Quality: {quality} (Type: {quality.dtype})")
            print(f"  View: {view} (Type: {view.dtype})")
            print(f"  Presence: {presence} (Type: {presence.dtype})")
            print("Data loading successful!")
        else:
            print("Dataset is empty. Check data paths.")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_data()
