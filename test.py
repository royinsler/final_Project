from datasets import load_dataset
import os
import random
from PIL import ImageDraw, ImageFont, Image
from torch import nn
from transformers import ViTFeatureExtractor, ResNetForImageClassification
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
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, ToPILImage, RandomResizedCrop, RandomVerticalFlip, RandomRotation, RandomHorizontalFlip
from transformers import BeitImageProcessor, BeitForImageClassification
from transformers import AutoFeatureExtractor, SwinForImageClassification, CvtForImageClassification
from torchvision import transforms

data = '/mnt/md0/royi/final_Project/Baseline_resnet/pngs/'
num_classes = 2
num_epochs = 50

# When feature)Extract = False, we finetune the complete model, else we only update the reshaped layer params.
# try this with feature_extract = False
feature_extract = False

batch_size = 8

# Data Augmentation and Normalization for training
# For Validation, we will only normalize the data
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data, x), data_transforms[x]) for x in ['train', 'val']}
# Dataloaders are created from the datasets
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

TMPDIR=/mnt/data/ pip install --cache-dir=$TMPDIR --build $TMPDIR torch