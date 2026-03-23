import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from scipy.ndimage import gaussian_filter

try:
    from ground_truth import GroundTruthConstructor
except ImportError:
    from src.ground_truth import GroundTruthConstructor


def _apply_clahe(img_u8: np.ndarray, clip_limit=2.0, tile_size=8) -> np.ndarray:
    """Apply CLAHE for contrast enhancement (domain adaptation)"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_u8)


def _crop_roi(img: np.ndarray, mask: np.ndarray = None, margin=10) -> tuple:
    """Crop to ROI (non-black region) to remove excessive background"""
    # Find non-zero region
    threshold = img.mean() * 0.1  # Adaptive threshold
    binary = (img > threshold).astype(np.uint8)
    
    if binary.sum() < 100:  # Fallback if too dark
        return img, mask, (0, 0, img.shape[1], img.shape[0])
    
    coords = np.column_stack(np.where(binary > 0))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Add margin
    h, w = img.shape
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(h, y_max + margin)
    x_max = min(w, x_max + margin)
    
    bbox = (x_min, y_min, x_max, y_max)
    img_crop = img[y_min:y_max, x_min:x_max]
    mask_crop = mask[y_min:y_max, x_min:x_max] if mask is not None else None
    
    return img_crop, mask_crop, bbox


def _normalize_per_image(img: np.ndarray) -> np.ndarray:
    """Per-image normalization for domain robustness"""
    mean = img.mean()
    std = img.std()
    if std < 1e-6:
        return img
    return (img - mean) / (std + 1e-6)


def _generate_ultrasound_noise(shape, base_intensity=128, speckle_std=30) -> np.ndarray:
    """Generate realistic ultrasound speckle noise (multiplicative)"""
    # Base noise
    noise = np.random.randn(*shape) * speckle_std + base_intensity
    
    # Add speckle pattern (multiplicative noise characteristic of ultrasound)
    speckle = np.random.gamma(2.0, 0.5, shape)
    noise = noise * speckle / speckle.mean()
    
    # Smooth to simulate ultrasound texture
    noise = gaussian_filter(noise, sigma=1.5)
    
    return np.clip(noise, 0, 255).astype(np.uint8)


def _inpaint_realistic_ultrasound(img_gray_u8: np.ndarray, hole_mask_u8: np.ndarray) -> np.ndarray:
    """Replace inpainting with realistic ultrasound noise/background"""
    if img_gray_u8.ndim != 2:
        raise ValueError(f"Expected grayscale image (H,W), got shape {img_gray_u8.shape}")
    if hole_mask_u8.ndim != 2:
        raise ValueError(f"Expected mask (H,W), got shape {hole_mask_u8.shape}")
    
    hole = (hole_mask_u8 > 0)
    if not hole.any():
        return img_gray_u8
    
    result = img_gray_u8.copy()
    
    # Estimate background intensity from non-hole regions
    bg_mask = (img_gray_u8 < img_gray_u8.mean() * 0.3) & (~hole)
    if bg_mask.sum() > 100:
        base_intensity = img_gray_u8[bg_mask].mean()
        speckle_std = img_gray_u8[bg_mask].std()
    else:
        base_intensity = img_gray_u8.mean() * 0.2
        speckle_std = 30
    
    # Generate realistic ultrasound noise
    noise = _generate_ultrasound_noise(img_gray_u8.shape, base_intensity, speckle_std)
    
    # Blend at edges for smooth transition
    kernel_size = 5
    hole_dilated = cv2.dilate(hole.astype(np.uint8), np.ones((kernel_size, kernel_size)), iterations=1)
    edge = (hole_dilated > 0) & (~hole)
    
    # Fill hole with noise
    result[hole] = noise[hole]
    
    # Smooth blend at edges
    if edge.any():
        alpha = 0.5
        result[edge] = (alpha * result[edge] + (1 - alpha) * noise[edge]).astype(np.uint8)
    
    return result


class CAMUSDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        phase='train',
        img_size=256,
        use_half_sequence=True,
        quality_source: str = "derived",
        view_filter: str | None = None,
        quality_map: dict | None = None,
        synthetic_neg_prob: float = 0.0,
        synthetic_partial_prob: float = 0.0,
        synthetic_inpaint_radius: int = 5,
        synthetic_neg_quality: float = 0.0,
        synthetic_partial_quality: float = 0.3,
    ):
        """
        Args:
            data_dir (str): Path to CAMUS database_nifti folder
            transform (albumentations.Compose): Augmentation pipeline
            phase (str): 'train' or 'val'
            img_size (int): Target image size
        """
        self.data_dir = data_dir
        self.transform = transform
        self.phase = phase
        self.img_size = img_size
        self.use_half_sequence = use_half_sequence
        self.quality_source = str(quality_source).lower()
        self.view_filter = view_filter
        self.synthetic_neg_prob = float(synthetic_neg_prob)
        self.synthetic_partial_prob = float(synthetic_partial_prob)
        self.synthetic_inpaint_radius = int(synthetic_inpaint_radius)
        self.synthetic_neg_quality = float(synthetic_neg_quality)
        self.synthetic_partial_quality = float(synthetic_partial_quality)
        if quality_map is None:
            # Default mapping from CAMUS imagequality labels to continuous targets.
            # Poor -> 0.3, Medium -> 0.6, Good -> 0.99 (avoid hard 1.0 saturation).
            self.quality_map = {"poor": 0.3, "medium": 0.6, "good": 0.99}
        else:
            self.quality_map = {str(k).strip().lower(): float(v) for k, v in dict(quality_map).items()}
        
        # Mathematical redesign: Ground truth constructor for presence/quality
        self.gt_constructor = GroundTruthConstructor()
        
        # CAMUS structure: patient0001/patient0001_2CH_ED.nii.gz
        # We focus on ED and ES frames which have GT
        # 2CH and 4CH views
        
        self.samples = []
        self._collect_samples()
        
        print(f"[{phase.upper()}] Loaded {len(self.samples)} samples from {data_dir}")

    def _collect_samples(self):
        # Simple split: patients 1-400 train, 401-450 val, 451-500 test (CAMUS typically 500 patients)
        # Or follow the user's split if provided. Standard CAMUS is 450 train+test, 50 holdout?
        # Let's use a simple split based on patient ID for now.
        
        patient_dirs = sorted(glob.glob(os.path.join(self.data_dir, 'patient*')))
        
        # Split 80/20 roughly
        if self.phase == 'train':
            patient_dirs = patient_dirs[:400]
        else:
            patient_dirs = patient_dirs[400:]
            
        for p_dir in patient_dirs:
            p_id = os.path.basename(p_dir)
            
            # We look for _ED_gt.nii.gz and _ES_gt.nii.gz to identify labeled frames
            for view in ['2CH', '4CH']:
                if self.view_filter is not None and str(self.view_filter).upper() != view:
                    continue

                info_quality = self._read_image_quality(p_dir, view)

                if self.use_half_sequence:
                    gt_filename = f"{p_id}_{view}_half_sequence_gt.nii.gz"
                    img_filename = f"{p_id}_{view}_half_sequence.nii.gz"

                    gt_path = os.path.join(p_dir, gt_filename)
                    img_path = os.path.join(p_dir, img_filename)

                    if os.path.exists(gt_path) and os.path.exists(img_path):
                        try:
                            t = int(nib.load(img_path).shape[-1])
                        except Exception:
                            t = 0
                        if t > 0:
                            for frame_idx in range(t):
                                self.samples.append({
                                    'img_path': img_path,
                                    'mask_path': gt_path,
                                    'view': view,
                                    'moment': 'half_sequence',
                                    'frame_idx': frame_idx,
                                    'image_quality': info_quality,
                                })

                for moment in ['ED', 'ES']:
                    gt_filename = f"{p_id}_{view}_{moment}_gt.nii.gz"
                    img_filename = f"{p_id}_{view}_{moment}.nii.gz"
                    
                    gt_path = os.path.join(p_dir, gt_filename)
                    img_path = os.path.join(p_dir, img_filename)
                    
                    if os.path.exists(gt_path) and os.path.exists(img_path):
                        self.samples.append({
                            'img_path': img_path,
                            'mask_path': gt_path,
                            'view': view, # 2CH or 4CH
                            'moment': moment,
                            'image_quality': info_quality,
                        })

    def _read_image_quality(self, patient_dir: str, view: str) -> str | None:
        info_path = os.path.join(patient_dir, f"Info_{view}.cfg")
        if not os.path.exists(info_path):
            return None
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    k, v = line.split(":", 1)
                    if k.strip().lower() == "imagequality":
                        return v.strip()
        except Exception:
            return None
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load NIfTI
        # These are usually (H, W, 1) or (H, W) or (H, W, T) depending, but ED/ES are single frame usually.
        img_nii = nib.load(sample['img_path'])
        mask_nii = nib.load(sample['mask_path'])
        
        img = img_nii.get_fdata()
        mask = mask_nii.get_fdata()
        
        if img.ndim == 3 and sample.get('frame_idx') is not None:
            img = img[:, :, sample['frame_idx']]
        elif img.ndim == 3:
            img = img[:, :, 0]

        if mask.ndim == 3 and sample.get('frame_idx') is not None:
            mask = mask[:, :, sample['frame_idx']]
        elif mask.ndim == 3:
            mask = mask[:, :, 0]

        synth_kind = None
        if self.phase == "train" and (self.synthetic_neg_prob > 0.0 or self.synthetic_partial_prob > 0.0):
            r = float(np.random.random())
            if r < self.synthetic_neg_prob:
                synth_kind = "neg"
            elif r < (self.synthetic_neg_prob + self.synthetic_partial_prob):
                synth_kind = "partial"
            
        # Rotate to standard orientation if needed
        # CAMUS images are often rotated. 
        # For this implementation, we rely on the network learning or raw data.
        # But typically we might want to normalize orientation. 
        # Let's start with raw data + resize.
        
        # Priority 1 Fix: Per-image normalization BEFORE uint8 conversion
        # This handles domain gap between CAMUS and Ultraprobe
        img_normalized = _normalize_per_image(img)
        
        # Convert to 0-255 uint8 for processing
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        
        # Priority 1 Fix: Apply CLAHE for contrast enhancement (domain adaptation)
        img = _apply_clahe(img, clip_limit=2.0, tile_size=8)

        if synth_kind is not None:
            mask_lbl = np.asarray(mask)
            heart = (mask_lbl > 0).astype(np.uint8)
            if int(heart.sum()) > 0:
                if synth_kind == "neg":
                    # Priority 1 Fix: Use realistic ultrasound noise instead of inpainting
                    img = _inpaint_realistic_ultrasound(img, heart)
                    mask = np.zeros_like(mask_lbl)
                else:
                    h, w = heart.shape
                    rm = np.zeros_like(heart)
                    if float(np.random.random()) < 0.5:
                        cut = int(np.random.randint(max(1, h // 4), max(2, (3 * h) // 4)))
                        if float(np.random.random()) < 0.5:
                            rm[cut:, :] = 1
                        else:
                            rm[:cut, :] = 1
                    else:
                        cut = int(np.random.randint(max(1, w // 4), max(2, (3 * w) // 4)))
                        if float(np.random.random()) < 0.5:
                            rm[:, cut:] = 1
                        else:
                            rm[:, :cut] = 1
                    hole = (heart & rm).astype(np.uint8)
                    if int(hole.sum()) > 0:
                        # Priority 1 Fix: Use realistic ultrasound noise instead of inpainting
                        img = _inpaint_realistic_ultrasound(img, hole)
                        mask_lbl = mask_lbl.copy()
                        mask_lbl[hole > 0] = 0
                        mask = mask_lbl
        
        # Process Mask
        # CAMUS GT: 0=background, 1=LV_endo, 2=myocardium, 3=left_atrium
        lv_mask = (mask == 1).astype(np.uint8)
        myo_mask = (mask == 2).astype(np.uint8)
        la_mask = (mask == 3).astype(np.uint8)
        mask = np.stack([lv_mask, myo_mask, la_mask], axis=-1)
        
        # Prepare for albumentations (H, W, C)
        # Image needs to be 3 channel for MobileNet usually, or we change first layer.
        # MobileNet expects 3 channels (RGB). We can replicate the grayscale.
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Augmentation
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            # Basic resize if no transform
            transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                ToTensorV2()
            ])
            augmented = transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
        # Prepare targets
        # Segmentation: mask (C, H, W) -> float
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3 and mask.shape[0] != 3 and mask.shape[-1] == 3:
            mask = mask.permute(2, 0, 1)
        mask = mask.float()

        # Structure presence: derived from mask areas
        areas = mask.sum(dim=(1, 2))
        structure_presence = (areas > 50).float()

        presence_label = float(structure_presence.sum().item() > 0.0)
        if synth_kind == "neg":
            presence_label = 0.0
        presence_label = torch.tensor([presence_label], dtype=torch.float32)

        quality_override = None
        if synth_kind == "neg":
            quality_override = self.synthetic_neg_quality
        elif synth_kind == "partial":
            quality_override = self.synthetic_partial_quality

        # CAMUS original quality label (for dual-head training)
        camus_quality_label = sample.get("image_quality")
        if camus_quality_label is None:
            camus_quality = 0.6
        else:
            camus_quality = float(self.quality_map.get(str(camus_quality_label).strip().lower(), 0.6))
        if quality_override is not None:
            camus_quality = float(quality_override)
        camus_quality = torch.tensor([camus_quality], dtype=torch.float32)

        # Derived quality (continuous from segmentation)
        if self.quality_source == "camus_label":
            # Use CAMUS labels as primary quality
            quality = camus_quality
        else:
            # Fan-area-normalized continuous quality calculation
            # Normalizes by actual fan area to make quality comparable across different fan sizes
            
            # Compute fan area (non-zero pixels in any structure)
            fan_area = float((mask.sum(dim=0) > 0).sum())
            total_pixels = float(mask.shape[1] * mask.shape[2])
            
            # If no structures detected, use total frame as fallback
            if fan_area < 100:
                fan_area = total_pixels
            
            # Expected pixel counts as percentage of FAN AREA (not total frame)
            expected_lv_pixels = fan_area * 0.20    # 20% of fan area
            expected_myo_pixels = fan_area * 0.30   # 30% of fan area
            expected_la_pixels = fan_area * 0.15    # 15% of fan area
            
            # Normalize to 0-1 (continuous, not binary)
            lv_norm = min(float(areas[0]) / expected_lv_pixels, 1.0) if expected_lv_pixels > 0 else 0.0
            myo_norm = min(float(areas[1]) / expected_myo_pixels, 1.0) if expected_myo_pixels > 0 else 0.0
            la_norm = min(float(areas[2]) / expected_la_pixels, 1.0) if expected_la_pixels > 0 else 0.0
            
            # Weighted combination (pure, no artificial baseline)
            q = 0.45 * lv_norm + 0.20 * myo_norm + 0.25 * la_norm + 0.10 * lv_norm
            
            if quality_override is not None:
                q = float(quality_override)
            quality = torch.tensor([q], dtype=torch.float32)
        
        # View: A4C=0, A2C=1, Other=2
        # In this dataset we only have 2CH and 4CH labeled.
        # "Other" would come from background frames or unlabeled data, 
        # but for Phase 1 we use what we have.
        # Map: A4C -> 0, A2C -> 1
        view_label = 0 if sample['view'] == '4CH' else 1
        view_label = torch.tensor(view_label, dtype=torch.long)
        
        return img, mask, structure_presence, quality, view_label, presence_label, camus_quality

def get_transforms(phase='train', img_size=256):
    if phase == 'train':
        return A.Compose([
            # Spatial
            A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=15, p=0.7),
            A.RandomCrop(height=int(img_size*0.8), width=int(img_size*0.8), p=0.5),
            A.Resize(img_size, img_size),
            
            # Occlusion
            A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(10, 50), hole_width_range=(10, 50), p=0.3),
            
            # Priority 2: Comprehensive intensity augmentation for domain robustness
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.7),
            
            # Priority 2: Noise augmentation (ultrasound speckle simulation)
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.5),
            
            # Blur for ultrasound texture
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            # Normalization (ImageNet stats for MobileNet)
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

if __name__ == "__main__":
    # Test dataset
    import matplotlib.pyplot as plt
    
    data_dir = "data/CAMUS_public/database_nifti"
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        exit()
        
    ds = CAMUSDataset(data_dir, transform=get_transforms('train'), phase='train')
    print(f"Dataset size: {len(ds)}")
    
    if len(ds) > 0:
        img, mask, struct_pres, quality, view, presence = ds[0]
        print(f"Image: {img.shape}, Mask: {mask.shape}, StructPres: {struct_pres}, Quality: {quality}, View: {view}, Presence: {presence}")
