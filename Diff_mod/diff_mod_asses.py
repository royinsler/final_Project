import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import inception_v3
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from math import log10
from scipy.linalg import sqrtm

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # For InceptionV3 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_images_from_path(path, max_images=None):
    """Load images from a given directory path."""
    image_list = []
    for idx, file in enumerate(os.listdir(path)):
        if max_images and idx >= max_images:
            break
        filepath = os.path.join(path, file)
        try:
            image = Image.open(filepath).convert('RGB')
            image_list.append(transform(image))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    return torch.stack(image_list)

def mse_metric(real, generated):
    """Calculate Mean Squared Error."""
    return torch.mean((real - generated) ** 2).item()

def psnr_metric(real, generated):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = mse_metric(real, generated)
    return 20 * log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

def ssim_metric(real, generated):
    """Calculate Structural Similarity Index (SSIM)."""
    real_np = real.numpy().transpose(1, 2, 0)
    generated_np = generated.numpy().transpose(1, 2, 0)
    return ssim(real_np, generated_np, multichannel=True, win_size=3, data_range=real_np.max() - real_np.min())

def compute_fid(real_activations, generated_activations):
    """Calculate Fréchet Inception Distance (FID)."""
    mu_real, sigma_real = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu_generated, sigma_generated = generated_activations.mean(axis=0), np.cov(generated_activations, rowvar=False)

    # Compute FID using the formula
    diff = mu_real - mu_generated
    covmean = sqrtm(sigma_real @ sigma_generated)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum(diff ** 2) + np.trace(sigma_real + sigma_generated - 2 * covmean)
    return fid

def get_inception_activations(images, model):
    """Get activations from the InceptionV3 model."""
    model.eval()
    with torch.no_grad():
        activations = model(images).detach().cpu().numpy()
    return activations

# Load real and generated images
real_path = "/mnt/data/tomosynthesis_data/fp_data/test"
generated_path = "/mnt/md0/royi/final_Project/seg_fp_synthetic_images_5000"
real_images = load_images_from_path(real_path, max_images=100)
generated_images = load_images_from_path(generated_path, max_images=100)

# Metrics calculations
mse_scores, psnr_scores, ssim_scores = [], [], []
for real, generated in zip(real_images, generated_images):
    mse_scores.append(mse_metric(real, generated))
    psnr_scores.append(psnr_metric(real, generated))
    ssim_scores.append(ssim_metric(real, generated))

print(f"Mean MSE: {np.mean(mse_scores):.4f}")
print(f"Mean PSNR: {np.mean(psnr_scores):.4f} dB")
print(f"Mean SSIM: {np.mean(ssim_scores):.4f}")

# FID Calculation
inception_model = inception_v3(pretrained=True, transform_input=False).eval()
inception_model.fc = torch.nn.Identity()  # Remove classification layer for activations

real_activations = get_inception_activations(real_images, inception_model)
generated_activations = get_inception_activations(generated_images, inception_model)

fid_score = compute_fid(real_activations, generated_activations)
print(f"FID Score: {fid_score:.4f}")
