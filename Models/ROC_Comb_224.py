import torch
import numpy as np
import os
from transformers import AutoImageProcessor, Swinv2Model, DefaultDataCollator, ResNetForImageClassification
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, ToPILImage, RandomResizedCrop, RandomVerticalFlip, RandomRotation, RandomHorizontalFlip
from datasets import load_dataset
from transformers import ViTFeatureExtractor, ViTForImageClassification
import pandas as pd
import random
import shutil
import torch.nn as nn
from transformers import BeitImageProcessor, BeitForImageClassification
from transformers import AutoFeatureExtractor, SwinForImageClassification
from transformers import AutoFeatureExtractor, SwinForImageClassification, CvtForImageClassification
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import seaborn as sns
from safetensors.torch import load_file
from transformers import ConvNextForImageClassification, ConvNextConfig
from transformers import BeitForImageClassification
from PIL import UnidentifiedImageError, Image

# Metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


pred_flag = True
evaluate_flag = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = "Swin-base-DBT-384-4-lr4"
device = 'cuda'
models = ['swin']#,'vit','res101']
model_data = {}

model_types = {
    # 'ViT': {
    #     'model_path': 'google/vit-base-patch16-224-in21k',
    #     'model_creator': ViTForImageClassification,
    #     'fine-tuned_model_patch': '/mnt/md0/royi/final_Project/ViT_output/checkpoint-13326/model.safetensors'
    # },
    'Swin': {
        'model_path': 'microsoft/swin-base-patch4-window7-224-in22k',
        'model_creator': SwinForImageClassification,
        'fine-tuned_model_patch': '/mnt/md0/royi/final_Project/idans models/pytorch_model.bin'
        # 'fine-tuned_model_patch': '/mnt/md0/royi/final_Project/Swin_output/checkpoint-12576/model.safetensors'

    },
    # 'ResNet': {
    #     'model_path': 'microsoft/resnet-101',
    #     'model_creator': ResNetForImageClassification,
    #     'fine-tuned_model_patch': '/mnt/md0/royi/final_Project/ResNet_output/checkpoint-4442/model.safetensors'
    # },
    # 'Convnext': {
    #     'model_path': 'facebook/convnext-base-224',
    #     'model_creator': ConvNextForImageClassification,
    #     'fine-tuned_model_patch': '/mnt/md0/royi/final_Project/Convnext_output/checkpoint-13326/model.safetensors'
    # },
    # 'BeiT': {
    #     'model_path': 'microsoft/beit-base-patch16-224',
    #     'model_creator': BeitForImageClassification,
    #     'fine-tuned_model_patch': '/mnt/md0/royi/final_Project/BeiT_output/checkpoint-6663/model.safetensors'
    # }
}

# Loop over each model type
for model_name, model_info in model_types.items():
    model_path = model_info['model_path']
    model_creator = model_info['model_creator']
    fine_tuned_model_path = model_info['fine-tuned_model_patch']

    # Load the model using the creator function

    print(f'Running {model_name}')


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


    # Load feature extractor and model
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    transform = Compose([ SafeResize((224, 224)),ToTensor(), normalize])
    test_data_path = "/mnt/data/tomosynthesis_data/labeled_data_224/Test"
    test_dataset = ImageFolder(test_data_path, transform=transform)
    batch_size = 16
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model
    labels = [0, 1]
    model = model_creator.from_pretrained(
        model_path,
        ignore_mismatched_sizes=True,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )
    model = model.to(device)
    # state_dict = load_file(fine_tuned_model_path)
    # # model.eval()
    # model.load_state_dict(state_dict)
    model.load_state_dict(torch.load(fine_tuned_model_path, weights_only=True))


    def sig(x):
        return 1 / (1 + np.exp(-x))


    def validate(model, loader):
        probabilities = []
        true_labels = []
        model.eval()  # set to eval mode to avoid batchnorm
        with torch.no_grad():  # avoid calculating gradients
            for images, labels in loader:
                img = images.to(device)
                p = sig(model(img).logits.cpu().numpy()[:, 1])
                probabilities.extend(p)
                true_labels.extend(labels)
        return probabilities, true_labels
        # Train
    probs, true = validate(model, test_dataloader)
    neg_files = os.listdir(test_data_path + "/Negative")
    pos_files = os.listdir(test_data_path + "/Positive")

    print(np.array(true).reshape(len(true), ).shape)
    print(np.array(probs).reshape(len(probs), ).shape)

    results_train = pd.DataFrame({"Filename": neg_files + pos_files,
                                  "true_labels": np.array(true).reshape(len(true), ),
                                  "Probabilities": np.array(probs).reshape(len(probs), )})
    results_train.to_csv(str("test_prediction-" + model_name + ".csv"), index=False)
    print('Test prediction done!')
# data

    from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

    print('--------------------     Model - ', model_name, '     --------------------')
    # Accuracy

    df = pd.read_csv(str("test_prediction-" + model_name + ".csv"))

    # JPGs eval
    predictions_jpg = np.where(df["Probabilities"] < 0.5, 0, 1)
    acc_jpg = accuracy_score(df["true_labels"], predictions_jpg)
    print('jpg-based Accuracy: ', acc_jpg)

    # Confusion matrix
    cm_jpg = confusion_matrix(df["true_labels"], predictions_jpg)
    print(cm_jpg)
    cm_jpg_percent = cm_jpg.astype('float') / cm_jpg.sum(axis=1)[:, np.newaxis] * 100
    labels = np.asarray([f'{v}\n({p:.2f}%)' for v, p in zip(cm_jpg.flatten(), cm_jpg_percent.flatten())]).reshape(2, 2)

    # Plot and save case-based confusion matrix as a JPG
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_jpg_percent, annot=labels, fmt='', cmap='Blues', cbar=False, xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'jpg_based Confusion Matrix {model_name}')
    plt.savefig(f'jpg_based_confusion_matrix_{model_name}.jpg')
    plt.savefig(f'jpg_based_confusion_matrix_{model_name}.jpg')
    plt.close()

    # sensitivity and specificity
    tn, fp, fn, tp = cm_jpg.ravel()
    sens = tp / (fn + tp)
    spec = tn / (tn + fp)
    print('JPG-based Sensitivity - ', sens)
    print('JPG-based Specificity - ', spec)

    # AUC
    auc_jpg = roc_auc_score(df["true_labels"], df["Probabilities"], average=None)
    print('JPG-based AUC - ', auc_jpg)

    # get the subjects and view list
    file_names = df["Filename"]
    uniqe_filename = []
    for name in file_names:
        if name[:12] not in uniqe_filename:
            uniqe_filename.append(name[:12])

    probabilities = []
    true_labels = []
    ten_slice_prob = []
    file_names = df["Filename"]
    #scan-based eval
    for name in uniqe_filename:
        res = file_names.str.contains(pat=name)

        sorted_list = sorted(zip(list(df["Filename"][res.values]), list(df["Probabilities"][res.values])), key=lambda x: int(x[0].split("_")[-1].split(".")[0]))
        sorted_list_probs = [item[1] for item in sorted_list]
        relevant_probs = np.array(sorted_list_probs)
        #relevant_probs = df["Probabilities"][res.values]
        y_true = df["true_labels"][res.values]
        probabilities.append(np.median(relevant_probs))
        true_labels.append(np.mean(y_true))

        maximum_prob = 0
        for idx in range(len(relevant_probs - 8)):
            current = np.mean(relevant_probs[idx:idx + 8])
            if current > maximum_prob:
                maximum_prob = current

        ten_slice_prob.append(maximum_prob)

    probabilities = np.asarray(probabilities).reshape(len(probabilities), )
    ten_slice_prob = np.asarray(ten_slice_prob).reshape(len(ten_slice_prob), )

    true_labels_scan = np.asarray(true_labels).reshape(len(true_labels), )
    predictions = np.where(probabilities < 0.5, 0, 1).reshape(len(probabilities), )
    ten_slice_prediction = np.where(ten_slice_prob < 0.5, 0, 1).reshape(len(ten_slice_prob), )

    results = pd.DataFrame({"Filename": uniqe_filename,
                            "true_labels": true_labels_scan,
                            "Predictions": ten_slice_prediction,
                            "Probabilities": ten_slice_prob})
    results.to_csv(f"results_test_scan-based_{model_name}.csv", index=False)


    # Accuracy
    predictions_scan = ten_slice_prediction
    probabilities_scan = ten_slice_prob
    acc = accuracy_score(true_labels_scan, predictions_scan)
    print('Scan-based Accuracy: ', acc)

    # Confusion matrix
    cm_scan = confusion_matrix(true_labels_scan, predictions_scan)
    cm_scan_percent = cm_scan.astype('float') / cm_scan.sum(axis=1)[:, np.newaxis] * 100
    labels = np.asarray([f'{v}\n({p:.2f}%)' for v, p in zip(cm_scan.flatten(), cm_scan_percent.flatten())]).reshape(2, 2)

    print(cm_scan)

    # Plot and save case-based confusion matrix as a JPG
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_scan_percent, annot=labels, fmt='', cmap='Blues', cbar=False, xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Scan-based Confusion Matrix')
    plt.savefig(f'scan_based_confusion_matrix_{model_name}.jpg')
    plt.close()

    # sensitivity and specificity
    tn, fp, fn, tp = cm_scan.ravel()
    sens = tp / (fn + tp)
    spec = tn / (tn + fp)
    print('Scan-based sensitivity - ', sens)
    print('Scan-based Specificity - ', spec)

    # AUC
    auc_scan = roc_auc_score(true_labels_scan, probabilities_scan, average=None)
    print('Scan-based AUC - ', auc_scan)
    fpr_scan, tpr_scan, thresholds = roc_curve(true_labels_scan, probabilities_scan)



    # fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
    # plt.plot(fpr, tpr, label=model_name_or_path + "AUC = " + str(round(auc, 4)))


    # subject based
    df = pd.read_csv(f"results_test_scan-based_{model_name}.csv")

    file_names = df["Filename"]

    # get the subjects and view list
    uniqe_filename = []
    for name in file_names:
        if name[:8] not in uniqe_filename:
            uniqe_filename.append(name[:8])

    probabilities = []
    true_labels = []
    ten_slice_prob = []
    file_names = df["Filename"]
    for name in uniqe_filename:
        res = file_names.str.contains(pat=name)
        relevant_probs = df["Probabilities"][res.values]
        y_true = df["true_labels"][res.values]
        # probabilities.append(np.mean(relevant_probs))
        probabilities.append(np.mean(relevant_probs))
        true_labels.append(np.mean(y_true))
    probabilities = np.asarray(probabilities).reshape(len(probabilities), )
    true_labels = np.asarray(true_labels).reshape(len(true_labels), )
    predictions = np.where(probabilities < 0.5, 0, 1).reshape(len(probabilities), )

    results = pd.DataFrame({"Filename": uniqe_filename,
                            "true_labels": true_labels,
                            "Predictions": predictions,
                            "Probabilities": probabilities})
    results.to_csv(f"results_test_case-based_{model_name}.csv", index=False)


    acc = accuracy_score(true_labels, predictions)
    print('Case-based Accuracy: ', acc)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    labels = np.asarray([f'{v}\n({p:.2f}%)' for v, p in zip(cm.flatten(), cm_percent.flatten())]).reshape(2, 2)

    print(cm)

    # Plot and save case-based confusion matrix as a JPG
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_percent, annot=labels, fmt='', cmap='Blues', cbar=False, xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Case-based Confusion Matrix {model_name}')
    plt.savefig(f'case_based_confusion_matrix_{model_name}.jpg')
    plt.savefig(f'case_based_confusion_matrix_{model_name}.jpg')
    plt.close()

    # sensitivity and specificity
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (fn + tp)
    spec = tn / (tn + fp)
    print('Case-based Sensitivity - ', sens)
    print('Case-based Specificity - ', spec)

    # AUC
    auc = roc_auc_score(true_labels, probabilities, average=None)
    print('Case-based AUC - ', auc)


    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
    # plt.plot(fpr, tpr, label="AUC = " + str(round(auc, 4)))


    model_data[model_name] = {
        'Scan-based': {
            'FPR': fpr_scan,
            'TPR': tpr_scan,
            'AUC': auc_scan,

        },
        'Case-based': {
            'FPR': fpr,
            'TPR': tpr,
            'AUC': auc,
        }
    }


case_names = ['Scan-based', 'Case-based']

# Initialize the plot
plt.figure(figsize=(8, 6))

# Loop through each case first
for case_name in case_names:
    # For each case, loop through the models
    for model_name, cases in model_data.items():
        print(model_name)
        FPR = cases[case_name]['FPR']
        TPR = cases[case_name]['TPR']
        AUC = cases[case_name]['AUC']
        # Plot ROC curve for the current case and model
        plt.plot(FPR, TPR, label=f'{model_name} - {case_name} (AUC = {AUC:.2f})')

    # Plot settings
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line for random classifier
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(case_name+' ROC Curves for Multiple Models Across')
    plt.legend(loc="lower right")
    plt.grid(True)

    # Show the plot
    # plt.tight_layout()
    # plt.show()
    plt.savefig(case_name+'_ROC.jpg')
    plt.close()
    plt.clf()