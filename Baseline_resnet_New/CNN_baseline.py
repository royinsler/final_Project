import device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.models import resnet101, ResNet101_Weights
import time

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Negative', 'Positive']  # Define your class names here
        self.file_list = []
        self.label_list = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for file in os.listdir(class_dir):
                self.file_list.append(os.path.join(class_dir, file))
                self.label_list.append(class_idx)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        image = Image.open(img_name).convert('RGB')
        label = self.label_list[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the image directories
train_dir = '/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train'  # Should contain 'Positive' and 'Negative' subdirectories
val_dir = '/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation'  # Should contain 'Positive' and 'Negative' subdirectories
test_dir = '/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Test'  # Should contain 'Positive' and 'Negative' subdirectories
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
torch.cuda.empty_cache()
# print(torch.cuda.memory_summary(device=None, abbreviated=False))



train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Create datasets and dataloaders
train_dataset = CustomDataset(root_dir=train_dir, transform=train_transform)
val_dataset = CustomDataset(root_dir=val_dir, transform=_transform)
test_dataset = CustomDataset(root_dir=test_dir)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Load a pretrained model, e.g., ResNet50
# base_model = models.resnet101(pretrained=True)
# in_features = base_model.fc.in_features
# base_model.fc = nn.Identity()


# Define the custom model
class CustomResNet(nn.Module):
    def __init__(self, num_classes=1):  # Binary classification (1 output)
        super(CustomResNet, self).__init__()
        self.base_model = resnet101(weights=ResNet101_Weights.DEFAULT)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.global_average_layer = nn.AdaptiveAvgPool2d((1, 1))
        self.prediction_layer = nn.Linear(in_features,num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        # x = self.global_average_layer(x)
        # x = torch.flatten(x, 1)
        x = self.prediction_layer(x)
        x = self.sigmoid(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# del variables
# gc.collect()
model = CustomResNet(num_classes=1).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
# Define a loss function
criterion = nn.BCELoss()  # For binary classification

# Define parameter groups with different learning rates
optimizer = optim.AdamW(model.parameters(), lr=0.001)


# Define a learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# Define early stopping and checkpoint
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


early_stopping = EarlyStopping(patience=10, verbose=True)
checkpoint_path = "model_checkpoint.pth"


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, early_stopping,
                checkpoint_path,device):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.float().view(-1, 1).to(device,
                                                                                                 non_blocking=True)
            optimizer.zero_grad()
            # print(inputs.shape)
            outputs = model(inputs)
            # labels = labels.float().view(-1, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += torch.sum(preds == labels.view_as(preds)).item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.float().view(-1, 1).to(device,
                                                                                                     non_blocking=True)
                outputs = model(inputs)
                # labels = labels.float().view(-1, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += torch.sum(preds == labels.view_as(preds)).item()
                val_total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

        early_stopping(val_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step()

    return train_losses, val_losses, train_accuracies, val_accuracies


# Train the model
train_flag = True
if train_flag:
    num_epochs = 50
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, early_stopping, checkpoint_path,device
    )

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.savefig('model_performance.jpg')
    plt.close()
    plt.clf()


# Test function
def test_model(model, test_loader, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)



# Test the model
test_flag = True
if test_flag:
    preds, labels, probs = test_model(model, test_loader, checkpoint_path,device)

    # Scan-based metrics
    accuracy = accuracy_score(labels, preds)
    print('Scan-based Accuracy: ', accuracy)

    cm = confusion_matrix(labels, preds)
    print('Confusion Matrix: \n', cm)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print('Scan-based Sensitivity: ',sensitivity)

def compute_statistics(outputs, labels):
    outputs = torch.sigmoid(outputs).numpy()
    labels = labels.numpy()
    fpr, tpr, thresholds = roc_curve(labels, outputs)
    auc = roc_auc_score(labels, outputs)
    return auc, fpr, tpr