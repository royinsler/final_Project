from datasets import load_dataset
import os
import random
from PIL import ImageDraw, ImageFont, Image
from transformers import ViTFeatureExtractor
import torch
import numpy as np
from datasets import load_metric
from transformers import ViTForImageClassification, AdamW
from transformers import Trainer
from transformers import TrainingArguments, EarlyStoppingCallback
import datasets
import transformers
import evaluate
from transformers import AutoImageProcessor, Swinv2Model, DefaultDataCollator
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, ToPILImage, RandomResizedCrop, \
    RandomVerticalFlip, RandomRotation, RandomHorizontalFlip
from transformers import BeitImageProcessor, BeitForImageClassification
from transformers import AutoFeatureExtractor, SwinForImageClassification, CvtForImageClassification
import PIL.Image


labels = [0, 1]

model_name_or_path = "microsoft/swin-base-patch4-window7-224-in22k"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
model = SwinForImageClassification.from_pretrained(
    model_name_or_path,
    ignore_mismatched_sizes=True,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

# ------------------------------------------ Data Functions ---------------------------------------------------
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
_train_transform = Compose([
    RandomHorizontalFlip(0.5),  # This is usually safe for medical images
    ToTensor(),
    normalize
])

print("Feature extractor parameters:")
print(f"Mean: {feature_extractor.image_mean}")
print(f"Std: {feature_extractor.image_std}")

def check_image_stats(img):
    # Convert PIL Image to numpy array and ensure it's numeric
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float32)

    print("Before preprocessing:")
    print(f"Shape: {img_array.shape}")
    print(f"Min: {float(img_array.min())}, Max: {float(img_array.max())}, Mean: {float(img_array.mean()):.2f}")

    # Check original channels
    print("\nOriginal channel means:")
    for i in range(3):
        channel_mean = float(img_array[:, :, i].mean())
        print(f"Channel {i}: Mean = {channel_mean:.2f}")

    # After preprocessing
    processed = _train_transform(img)
    print("\nAfter preprocessing:")
    print(f"Shape: {processed.shape}")
    print(f"Min: {float(processed.min())}, Max: {float(processed.max())}, Mean: {float(processed.mean()):.2f}")

    # Check processed channels
    print("\nProcessed channel means:")
    for i in range(3):
        channel_mean = float(processed[i].mean())
        print(f"Channel {i}: Mean = {channel_mean:.2f}")

    # Check channel differences
    diff_0_1 = float((processed[0] - processed[1]).abs().max())
    diff_1_2 = float((processed[1] - processed[2]).abs().max())
    print(f"\nMaximum absolute differences between channels:")
    print(f"Channels 0-1: {diff_0_1:.6f}")
    print(f"Channels 1-2: {diff_1_2:.6f}")

image_path = '/mnt/data/tomosynthesis_data/labeled_data_224/Test/Negative/AA3345_L_CC_8.png'
check_image_stats(image_path)