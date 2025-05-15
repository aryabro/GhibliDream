import os
import numpy as np
from PIL import Image 
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.linalg import sqrtm 
import torch.nn as nn

ORIGINAL_IMAGES_FOLDER = './original_images'
GENERATED_IMAGES_FOLDER = './generated_images'

# Target image size for InceptionV3 (must be 299x299)
TARGET_IMG_SIZE = (299, 299)
BATCH_SIZE = 32


class ImageDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for loading images from a folder.
    Resizes images to 1024x1024 (original) or 512x512 (generated) then to 299x299 for Inception.
    """
    def __init__(self, folder, target_inception_size, original_image_size_before_inception, transform=None):
        self.folder = folder
        self.transform = transform 
        self.target_inception_size = target_inception_size
        self.original_image_size_before_inception = original_image_size_before_inception 

        self.image_files = sorted([os.path.join(folder, f)
                                   for f in os.listdir(folder)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not self.image_files:
            raise ValueError(f"No image files found in folder: {folder}")

        valid_images = []
        for img_path in self.image_files:
            try:
                img = Image.open(img_path)
                img.verify() 
                valid_images.append(img_path)
            except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
                print(f"Warning: Skipping corrupted or invalid image {img_path}: {e}")
        self.image_files = valid_images

        if not self.image_files:
            raise ValueError(f"No valid images could be found after integrity check in folder: {folder}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if image.size != self.original_image_size_before_inception:
                 image = image.resize(self.original_image_size_before_inception, Image.Resampling.LANCZOS)

            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading or transforming image {img_path} during __getitem__: {e}")
            return None

def collate_fn_filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: 
        return None
    return torch.utils.data.dataloader.default_collate(batch)

class InceptionV3FeatureExtractor(nn.Module):
    """
    Wrapper for InceptionV3 to extract features for FID.
    """
    def __init__(self):
        super().__init__()
        self.inception_v3 = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1
        )
        self.inception_v3.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inception_v3.eval() 
        features = self.inception_v3(x)
        if features.ndim == 4 and features.shape[2] == 1 and features.shape[3] == 1:
            features = features.squeeze(3).squeeze(2)
        return features

def get_image_loader(folder_path, original_image_size, target_inception_size, batch_size):
    """
    Creates a DataLoader for images in a folder.
    Uses the globally defined ImageDataset and collate_fn_filter_none.
    """
    inception_transform = transforms.Compose([
        transforms.Resize(target_inception_size, interpolation=transforms.InterpolationMode.LANCZOS), # to 299x299
        transforms.ToTensor(),                # Converts to [0, 1] and C x H x W
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    ])

    dataset = ImageDataset(folder_path, target_inception_size, original_image_size, transform=inception_transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn_filter_none,
        pin_memory=False
    )
    return loader


@torch.no_grad()
def calculate_activations_pytorch(loader, model):
    """
    Calculates InceptionV3 activations for all images provided by the DataLoader.
    """
    model.eval()

    all_activations = []
    for batch in loader:
        if batch is None: 
            print("Warning: Skipping an entire batch because all images in it failed to load/transform.")
            continue
        activations = model(batch)
        all_activations.append(activations.cpu().numpy()) 

    if not all_activations:
        return np.empty((0, 2048)) 

    return np.concatenate(all_activations, axis=0)


def calculate_fid(act1: np.ndarray, act2: np.ndarray) -> float:
    """
    Calculates the Fréchet Inception Distance (FID) between two sets of activations.
    act1: activations from real images (NumPy array)
    act2: activations from generated images (NumPy array)
    """
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2.0)
    offset = np.eye(sigma1.shape[0]) * 1e-6
    covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == '__main__':

    print("Loading InceptionV3 model for feature extraction...")
    try:
        inception_model = InceptionV3FeatureExtractor()
        inception_model.eval()  
        print("InceptionV3 model loaded successfully.")
    except Exception as e:
        print(f"Fatal Error: Could not load InceptionV3 model: {e}")
        print("Ensure PyTorch and TorchVision are installed correctly and an internet connection is available for weight download.")
        exit()

    print(f"Loading original images from: {ORIGINAL_IMAGES_FOLDER}")
    try:
        original_loader = get_image_loader(ORIGINAL_IMAGES_FOLDER, (1024, 1024), TARGET_IMG_SIZE, BATCH_SIZE)
        if len(original_loader.dataset.image_files) < 2:
             raise ValueError("Not enough valid original images found to calculate FID (minimum 2 required).")
        print(f"DataLoader created for original images (initial valid file count: {len(original_loader.dataset.image_files)}).")
    except Exception as e:
        print(f"Fatal Error: Could not create DataLoader for original images: {e}")
        exit()

    print(f"Loading generated images from: {GENERATED_IMAGES_FOLDER}")
    try:
        generated_loader = get_image_loader(GENERATED_IMAGES_FOLDER, (512, 512), TARGET_IMG_SIZE, BATCH_SIZE)
        if len(generated_loader.dataset.image_files) < 2:
            raise ValueError("Not enough valid generated images found to calculate FID (minimum 2 required).")
        print(f"DataLoader created for generated images (initial valid file count: {len(generated_loader.dataset.image_files)}).")
    except Exception as e:
        print(f"Fatal Error: Could not create DataLoader for generated images: {e}")
        exit()

    print("Calculating activations for original images...")
    activations_original = calculate_activations_pytorch(original_loader, inception_model)
    num_original_processed = activations_original.shape[0]
    print(f"Original activations calculated. Shape: {activations_original.shape} (Successfully processed {num_original_processed} original images)")

    print("Calculating activations for generated images...")
    activations_generated = calculate_activations_pytorch(generated_loader, inception_model)
    num_generated_processed = activations_generated.shape[0]
    print(f"Generated activations calculated. Shape: {activations_generated.shape} (Successfully processed {num_generated_processed} generated images)")

    if num_original_processed < 2:
        print(f"Fatal Error: Not enough original image activations to calculate FID (need at least 2, got {num_original_processed}). Review image loading warnings.")
        exit()
    if num_generated_processed < 2:
        print(f"Fatal Error: Not enough generated image activations to calculate FID (need at least 2, got {num_generated_processed}). Review image loading warnings.")
        exit()

    if activations_original.shape[1] != 2048 or activations_generated.shape[1] != 2048:
        print(f"Fatal Error: Expected 2048 features from InceptionV3, but got {activations_original.shape[1]} for original and/or {activations_generated.shape[1]} for generated.")
        exit()

    print("Calculating FID score...")
    try:
        fid_score = calculate_fid(activations_original, activations_generated)
        print(f"------------------------------------")
        print(f"FID Score: {fid_score:.3f}")
        print(f"------------------------------------")
    except Exception as e:
        print(f"Error calculating FID score: {e}")
        if "singular matrix" in str(e).lower() or "division by zero" in str(e).lower() or "ształt" in str(e).lower() : # "ształt" is shape in Polish, sometimes appears in linalg errors
            print("This error often occurs if images in a set are too similar or identical, or if there are too few unique images, leading to a singular covariance matrix.")
            print("Ensure your image sets have diversity. A small offset was added to covariance matrices for stability, but severe issues persist with problematic data.")

    print("FID calculation finished.")