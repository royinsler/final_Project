import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from datasets import load_dataset
from diffusers import StableDiffusionImg2ImgPipeline

print("code version 1.0.9")

# Load the Stable Diffusion image-to-image pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.to("cuda")

# Dataset path
dataset_path = "/mnt/data/tomosynthesis_data/fp_data"

# Load dataset
dataset = load_dataset(
    "imagefolder",
    data_files={
        "test": os.path.join(dataset_path, "test/**")
    },
    drop_labels=False
)

import random

def generate_images_from_dataset(dataset, split='test', num_images=5, output_dir='fp_synthetic_images_5000'):
    os.makedirs(output_dir, exist_ok=True)

    # Convert dataset to list and shuffle
    samples = list(dataset[split])
    random.shuffle(samples)  # Shuffle the dataset

    # Process a limited number of samples
    for i, example in enumerate(samples[:num_images]):
        original_image = example['image']  # Access the shuffled image
        original_image = original_image.resize((512, 512))  # Resize to 512x512

        # Define a prompt
        prompt = (
            "A grayscale breast tissue image that is a part of volumetric DBT z-Stack of from either mediolateral oblique or Craniocaudal views.This image should have dense breast structures and calcifications, as well as realistic details that match medical imaging standards."
        )

        # Pass the image to the pipeline
        result = pipe(
            prompt=prompt,
            image=original_image,
            strength=0.05,
            guidance_scale=3
        )

        # Check NSFW flag
        if result["nsfw_content_detected"] and any(result["nsfw_content_detected"]):
            print(f"Skipping image {i + 1} due to NSFW detection.")
            continue  # Skip saving this image

        # Save the generated image
        generated_image = result.images[0]
        save_path = os.path.join(output_dir, f"fp_synthetic_image_{i + 1}.png")
        generated_image.save(save_path)
        print(f"Generated synthetic image saved to {save_path}")


# Run the function to generate synthetic images
generate_images_from_dataset(dataset, split='test', num_images=5000)
