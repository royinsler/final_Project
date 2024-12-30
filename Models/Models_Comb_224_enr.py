# Import Libraries
from random import random

from datasets import load_dataset
import os

from numpy.ma.core import resize
from transformers import AutoFeatureExtractor, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import ViTForImageClassification, SwinForImageClassification, ResNetForImageClassification,ViTImageProcessor
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip
import evaluate
import matplotlib.pyplot as plt
import transformers
from transformers import TrainerCallback
from transformers import DeiTForImageClassification
from transformers import ConvNextForImageClassification, ConvNextConfig
from transformers import BeitForImageClassification
# from transformers import NestForImageClassification
from PIL import UnidentifiedImageError, Image

# Parameters
gpu_number = 0
# dataset_path = "/mnt/md0/royi/final_Project"
dataset_path = "/mnt/data/tomosynthesis_data/labeled_data_224"
labels = [0, 1]
batch_size = 16
lr = 1e-4
# Python random
# random.seed('42')

# Numpy
# np.random.seed('42')
#
# # PyTorch
# torch.manual_seed('42')
# torch.cuda.manual_seed('42')
# torch.cuda.manual_seed_all('42')  # for multi-GPU
#
# # Additional CUDA settings for deterministic operations
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

#set custom cache dear to acomodate the server's structure
custom_cache_dir = "/mnt/data/HF/.cache"

clsss_proportion = len(os.listdir(os.path.join(dataset_path, "Train/Negative"))) / len(
    os.listdir(os.path.join(dataset_path, "Train/Positive")))
# Define model paths and names
model_types = {
    # 'ViT': {
    #     'model_path': 'google/vit-base-patch16-224-in21k',
    #     'model_creator': ViTForImageClassification
    # },
    # 'Swin': {
    #     'model_path': 'microsoft/swin-base-patch4-window7-224-in22k',
    #     'model_creator': SwinForImageClassification
    # },
    # 'ResNet': {
    #     'model_path': 'microsoft/resnet-101',
    #     'model_creator': ResNetForImageClassification
    # },
    'Convnext': {
        'model_path': 'facebook/convnext-base-224',
        'model_creator': ConvNextForImageClassification
    },
    'BeiT': {
        'model_path': 'microsoft/beit-base-patch16-224',
        'model_creator': BeitForImageClassification
    }
}

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

# Load dataset
dataset = load_dataset("imagefolder", data_files={"train": os.path.join(dataset_path, "Train/**"),
                                                  "valid": os.path.join(dataset_path, "Val/**")},
                       drop_labels=False, cache_dir=custom_cache_dir)

class SafeResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        try:
            # Resize the image
            # print("trying to resize")
            img = transforms.Resize(self.size)(img)
            return img
        except (UnidentifiedImageError, OSError) as e:
            print(f"Skipping image due to error: {e}")
            return None

# Define transforms
def get_transforms(feature_extractor):
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    _train_transform = Compose([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                SafeResize((224, 224)),  # Custom resizing with error handling
                                RandomVerticalFlip(0.5),
                                RandomRotation(10),
                                RandomHorizontalFlip(0.5),
                                ToTensor(),
                                normalize])
    _transforms = Compose([ToTensor(), normalize])
    return _train_transform, _transforms

# Data functions
def train_transform(examples, train_transforms):
    examples["pixel_values"] = [train_transforms(img) for img in examples["image"]]
    examples['labels'] = examples['label']
    del examples["image"]
    return examples

def transform(examples, eval_transforms):
    examples["pixel_values"] = [eval_transforms(img) for img in examples["image"]]
    examples['labels'] = examples['label']
    del examples["image"]
    return examples

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch]),
    }

# Evaluation metrics
def compute_metrics(p):
    metric1 = evaluate.load("accuracy")
    metric2 = evaluate.load("precision")
    metric3 = evaluate.load("recall")
    metric4 = evaluate.load("roc_auc")

    accuracy = metric1.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["accuracy"]
    precision = metric2.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["precision"]
    recall = metric3.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)["recall"]
    auc = metric4.compute(references=p.label_ids, prediction_scores=np.argmax(p.predictions, axis=1))["roc_auc"]
    return {"accuracy": accuracy, "PPV": precision, "sensitivity": recall, "roc_auc": auc}

# Loop over each model type
for model_name, model_info in model_types.items():
    model_path = model_info['model_path']
    model_creator = model_info['model_creator']

    # Load the model using the creator function
    model = model_creator.from_pretrained(
        model_path,
        ignore_mismatched_sizes=True,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )
    print(f'Running {model_name}')

    # Load feature extractor and model
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)


    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000,
                                                             num_training_steps=60000)
    optimizers = optimizer, scheduler

    # Get transforms
    _train_transform, _transforms = get_transforms(feature_extractor)

    # Prepare datasets
    prepared_ds = dataset["train"].with_transform(lambda examples: train_transform(examples, _train_transform))
    prepared_ds_val = dataset["valid"].with_transform(lambda examples: transform(examples, _transforms))

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./{model_name}_output",
        per_device_train_batch_size=batch_size,
        eval_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=10,
        fp16=True,
        save_strategy="epoch",
        learning_rate=lr,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )





    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_loss_history = []
        # def training_step(self, model, inputs):
        #     # Compute the loss using your custom compute_loss
        #     loss = self.compute_loss(model, inputs)
        #     self.epoch_train_loss += loss.item()  # Accumulate the training loss for the epoch
        #     self.steps_in_epoch += 1
        #     return loss

        # def on_epoch_end(self):
        #     # Calculate and log the average train loss for the epoch
        #     avg_train_loss = self.epoch_train_loss / self.steps_in_epoch
        #     self.train_loss.append(avg_train_loss)
        #     # Reset for the next epoch
        #     self.epoch_train_loss = 0
        #     self.steps_in_epoch = 0

        def evaluation_loop(self, *args, **kwargs):
            output = super().evaluation_loop(*args, **kwargs)
            eval_loss = output.metrics['eval_loss'] if 'eval_loss' in output.metrics else None
            if eval_loss is not None:
                self.eval_loss_history.append(eval_loss)  # Save eval loss after evaluation
            return output

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, clsss_proportion]).to("cuda"))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            # Log the loss for plotting
            # self.state.log_history.append({"loss": loss.item()})
            return (loss, outputs) if return_outputs else loss



    # Feature extractor for all models (ViT, Swin, ResNet)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

    # Initialize the trainer with the CustomTrainer class
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds,
        eval_dataset=prepared_ds_val,
        tokenizer=feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        optimizers=optimizers,
    )

    # Train the model
    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    # Get the training loss per epoch from the metrics
    train_loss_history = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    # Extract loss and epoch values
    # epochs = []
    # train_loss = []
    # Loop through the log history
    # for log_entry in trainer.state.log_history:
    print(trainer.eval_loss_history)
    print(train_loss_history)

    # if 'loss' in log_entry:  # Check if the entry contains loss data
    #     train_loss.append(log_entry['loss'])
    #     print(str(log_entry['epoch']))
    #     # If the entry contains epoch information, use it. Otherwise, use the global_step.
    #     # if 'epoch' in log_entry:
    #     epochs.append(log_entry['epoch'])
    # for log_entry in trainer.state.log_history:
    #     print(str(log_entry['loss']))
    #     # if 'loss' in log_entry:  # Check if the entry contains loss data
    #     train_loss.append(log_entry['loss'])
    #     print(str(log_entry['epoch']))
    #     # If the entry contains epoch information, use it. Otherwise, use the global_step.
    #     # if 'epoch' in log_entry:
    #     epochs.append(log_entry['epoch'])
            # else:
            #     epochs.append(len(epochs) + 1)  # If no epoch info, just increment
            #     print(len(epochs) + 1)
    # Plot Loss Curve
    # epochs = range(1, 20)
    epochs = list(range(1, len(train_loss_history) + 1))
    # Plot the training and evaluation loss
    plt.plot(epochs,trainer.eval_loss_history, label=f'{model_name} Eval Loss')
    plt.plot(epochs,train_loss_history, label=f'{model_name} Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


    # Evaluate the model on validation set
    metrics = trainer.evaluate(prepared_ds_val)
    print(f'Evaluation Metrics for {model_name}:', metrics)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

plt.title(f'Training and validation Losses for Models 224 res with enriched db')
plt.savefig(f'Loss_along_epochs_all_models_224_enr.jpg')