
#Initiailize Cuda
import os

from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.cuda.empty_cache()
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
CUDA_LAUNCH_BLOCKING=1
### Model and Training Code

import time
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms, models
from torchvision.models import resnet101, ResNet101_Weights
import seaborn as sns
from sklearn.metrics import classification_report
import timm


# Define the model with global average pooling and prediction layers
class ViT(nn.Module):
    def __init__(self, num_classes=1):
        super(ViT, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Define the train and evaluation functions
def train_model(model, dataloaders, criterion, optimizer, scheduler, class_proportions ,num_epochs, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # print(inputs)
                # print(labels)
                inputs = inputs.to(device)
                labels = labels.to(device).float().view(-1, 1)

                # print((labels[0:20]))

                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(outputs)
                    probs = torch.sigmoid(outputs) >= 0.5
                    # print(probs)
                    preds = probs >= 0.5  # Directly use the boolean tensor
                    # print(preds)
                    # loss_fct = criterion()
                    # loss = loss_fct(torch.sigmoid(outputs), labels)
                    loss = criterion(outputs, labels)
                    # class_proportions = class_proportions.to(device)
                    # weighted_loss = loss * class_proportions   # Apply weights
                    # loss = weighted_loss.mean()  # Take the mean of the weighted losses

                    # loss.requires_grad = True

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                # print(torch.sum(probs == labels.data))
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
            print(len(dataloaders[phase].dataset))
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Evaluate the model
def evaluate_model(model, dataloaders, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    class_names = ['Negative', 'Positive']

    all_labels = []
    all_preds = []
    all_probs = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            # print(outputs)
            probs = torch.sigmoid(outputs) >= 0.5
            preds = probs >= 0.5  # Directly use the boolean tensor
            # print(preds)

            # outputs = model(inputs)
            # outputs = torch.clamp(outputs, 0, 1)  # Clamp values between 0 and 1
            # probs = torch.sigmoid(outputs)
            # preds = probs.round()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            loss = criterion(torch.sigmoid(outputs), labels.float().view(-1, 1))

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data.view(-1, 1))

    test_loss = running_loss / len(dataloaders['test'].dataset)
    test_acc = running_corrects.double() / len(dataloaders['test'].dataset)

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    all_labels = np.array(all_labels).reshape(-1)
    all_probs = np.array(all_probs).reshape(-1)

    # Calculate AUC
    auc = roc_auc_score(all_labels, all_probs)
    print(f'AUC: {auc:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # Sensitivity and Specificity
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (fn + tp)
    spec = tn / (tn + fp)
    print(f'Sensitivity: {sens:.4f}')
    print(f'Specificity: {spec:.4f}')

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.savefig('roc_curve.png')
    plt.close()

    # Scan-based probabilities
    file_names = pd.Series(os.listdir(os.path.join(test_dir, 'Negative')) + os.listdir(os.path.join(test_dir, 'Positive')))
    probs = pd.Series(all_probs)
    # preds = pd.Series(all_preds)
    true = pd.Series(all_labels)

    unique_filename = []
    for name in file_names:
        if name[:12] not in unique_filename:
            unique_filename.append(name[:12])

    probabilities = []
    true_labels = []
    ten_slice_prob = []
    unique_filename_new = []
    for name in unique_filename:
        res = file_names.str.contains(pat=name)
        relevant_probs = probs[res]
        y_true = true[res]
        # print(y_true)[0:5]
        if len(y_true) == 0 or np.isnan(y_true).any():
            continue  # Skip this entry if y_true is empty or contains NaN
        else:
            unique_filename_new.append(name)
        # print(np.mean(y_true))
        true_label = y_true.mode()[0]  # Take the most frequent label (mode)
        true_labels.append(np.mean(true_label))
        maximum_prob = 0
        for idx in range(len(relevant_probs) - 8):
            current = np.mean(relevant_probs[idx:idx + 8])
            if current > maximum_prob:
                maximum_prob = current

        ten_slice_prob.append(maximum_prob)

    ten_slice_prob = np.asarray(ten_slice_prob).reshape(len(ten_slice_prob),)
    true_labels = np.asarray(true_labels).reshape(len(true_labels),)
    ten_slice_prediction = np.where(ten_slice_prob < 0.5, 0, 1).reshape(len(ten_slice_prob),)

    results = pd.DataFrame({"Filename": unique_filename_new,
                            "true_labels": true_labels,
                            "Predictions": ten_slice_prediction,
                            "Probabilities": ten_slice_prob})
    results.to_csv("results_test_scan-based.csv", index=False)
    # Accuracy
    predictions = ten_slice_prediction
    probabilities = ten_slice_prob
    acc = accuracy_score(true_labels, predictions)
    print('Scan-based Accuracy: ', acc)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plot and save subject-based confusion matrix as a JPG
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Scan-based Confusion Matrix')
    plt.savefig('scan_based_confusion_matrix.jpg')
    plt.close()

    # sensitivity and specificity
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (fn + tp)
    spec = tn / (tn + fp)
    print('Scan-based sensitivity - ', sens)
    print('Scan-based Specificity - ', spec)

    # AUC
    auc = roc_auc_score(true_labels, probabilities, average=None)
    print('Scan-based AUC - ', auc)

    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
    plt.plot(fpr, tpr, label="AUC = " + str(round(auc, 4)))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.savefig('scan_based_ROC.jpg')
    plt.close()
    plt.clf()
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

    # Subject-based probabilities
    df = pd.read_csv("results_test_scan-based.csv")

    file_names = df["Filename"]

    unique_filename = []
    for name in file_names:
        if name[:8] not in unique_filename:
            unique_filename.append(name[:8])

    probabilities = []
    true_labels = []
    file_names = df["Filename"]
    unique_filename_new = []
    for name in unique_filename:
        res = file_names.str.contains(pat=name)
        relevant_probs = df["Probabilities"][res]
        y_true = df["true_labels"][res]
        if np.isnan(y_true).any():
            unique_filename_new.append(name)
            continue  # Skip this entry if y_true is empty or contains NaN

        probabilities = np.append(probabilities, np.mean(relevant_probs))
        true_label = y_true.mode()[0]  # Take the most frequent label (mode)
        true_labels.append(np.mean(true_label))
        probabilities = np.asarray(probabilities).reshape(len(probabilities),)
    true_labels = np.asarray(true_labels).reshape(len(true_labels), )

    # Calculate subject-based accuracy, AUC, and other metrics
    predictions = np.where(probabilities < 0.5, 0, 1).reshape(len(probabilities), )

    subject_acc = accuracy_score(true_labels, predictions)
    print('Subject-based Accuracy: ', subject_acc)

    # Confusion matrix
    cm_subject = confusion_matrix(true_labels, predictions)
    print('Subject-based Confusion Matrix:')

    # Plot and save subject-based confusion matrix as a JPG
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_subject, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Subject-based Confusion Matrix')
    plt.savefig('subject_based_confusion_matrix.jpg')
    plt.close()

    # Sensitivity and Specificity
    tn, fp, fn, tp = cm_subject.ravel()
    sens_subject = tp / (fn + tp)
    spec_subject = tn / (tn + fp)
    print('Subject-based Sensitivity: ', sens_subject)
    print('Subject-based Specificity: ', spec_subject)

    # AUC
    auc_subject = roc_auc_score(true_labels, probabilities)
    print('Subject-based AUC: ', auc_subject)

    # ROC Curve
    fpr_subject, tpr_subject, _ = roc_curve(true_labels, probabilities)
    plt.plot(fpr_subject, tpr_subject, label=f"AUC = {auc_subject:.4f}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.savefig('subject_based_ROC.jpg')
    plt.close()# Adjust the dataloaders to return filenames
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Negative', 'Positive']  # Define your class names here
        self.file_list = []
        self.label_list = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if self.root_dir == '/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train':
                for file in os.listdir(class_dir)[0:5000]:
                    self.file_list.append(os.path.join(class_dir, file))
                    self.label_list.append(class_idx)
            else:
                for file in os.listdir(class_dir)[0:1500]:
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


def create_dataloaders(train_dir, val_dir, test_dir, batch_size, num_workers):
    train_transform = transforms.Compose([
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # Create datasets and dataloaders
    train_dataset = CustomDataset(root_dir=train_dir, transform=train_transform)
    val_dataset = CustomDataset(root_dir=val_dir, transform=val_transform)
    test_dataset = CustomDataset(root_dir=test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}


# Print dataset statistics
def dataset_statistics(dataloader, name):
    print("Loading "+ name+' Dataset')
    total = len(dataloader.dataset)
    pos = sum(y for _, y in dataloader.dataset)
    neg = total - pos
    print(f"{name} dataset: {total} samples, {pos} positive, {neg} negative")

# Main execution
if __name__ == '__main__':
    # Define the image directories
    train_dir = '/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train'  # Should contain 'Positive' and 'Negative' subdirectories
    val_dir = '/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Validation'  # Should contain 'Positive' and 'Negative' subdirectories
    test_dir = '/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Test'  # Should contain 'Positive' and 'Negative' subdirectories
    batch_size = 32
    num_workers = 4
    num_epochs = 100
    learning_rate = 0.01

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create dataloaders
    dataloaders = create_dataloaders(train_dir, val_dir, test_dir, batch_size, num_workers)
    # Initialize the model

    dataset_statistics(dataloaders['train'], "Training")
    dataset_statistics(dataloaders['val'], "Validation")
    dataset_statistics(dataloaders['test'], "Test")

    model = ViT()
    model = model.to(device)

    negative_count = len(os.listdir(os.path.join(train_dir, "Negative")))
    positive_count = len(os.listdir(os.path.join(train_dir, "Positive")))
    total_count = negative_count + positive_count
    negative_proportion = negative_count / total_count
    positive_proportion = positive_count / total_count
    class_proportions = torch.tensor([negative_proportion, positive_proportion])

    # Freeze the convolutional layers
    # for param in model.parameters():
    #     param.requires_grad = False

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.96)

    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, class_proportions, num_epochs=num_epochs, device=device)

    # Evaluate the model
    evaluate_model(model, dataloaders, criterion, device=device)


