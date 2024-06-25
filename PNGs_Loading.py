import warnings

warnings.filterwarnings("ignore")
import os
import numpy as np
import torch
import torchvision
from typing import Tuple, List
import matplotlib.pyplot as plt

LABELS_ENCODING = {
                    'Negative': 0,
                    'Positive': 1
                  }

def load_saved_labeled_data(path: str,
                            plot_example_labels: bool = False) -> Tuple[List, List]:
    # Loading and Labeling the data
    labels = []
    images = []
    for current_dir in os.listdir(path):
        if 'desktop' not in current_dir:
            print(f'Loading {current_dir}')
            for i, tr in enumerate(os.listdir(path + current_dir)):
                try:
                    img = torchvision.io.read_image(path=path + current_dir + "/" + tr,
                                                    mode=torchvision.io.ImageReadMode.GRAY)
                    img = torchvision.transforms.functional.resize(img, size=[64, 64])
                    labels.append(current_dir)
                    images.append(img)
                    if plot_example_labels and i == 0:
                        print(f'{current_dir} Example Image:')
                        plt.imshow(img.permute(1, 2, 0), cmap='gist_gray')
                        plt.show()

                except:
                    if 'desktop' not in tr:
                        print(f'Failed to load: {tr}')

    return torch.stack(images), torch.stack(convert_labels(labels))


def convert_labels(labels: List) -> List:
    return [torch.tensor([LABELS_ENCODING[label]]) for label in labels]









