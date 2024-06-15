import warnings

from torchvision.utils import save_image

warnings.filterwarnings("ignore")
import os
import numpy as np
import torch
import torchvision
from typing import Tuple, List
import matplotlib.pyplot as plt
import gc

LABELS_ENCODING = {
                    'Negative': 0,
                    'Positive': 1
                  }

def resize_labeled_data(path: str, size: int,
                            plot_example_labels: bool = False):
    # Loading and Labeling the data
    labels = []
    # images = []
    for current_dir in os.listdir(path):
        #if 'desktop' not in current_dir:
        print(f'Loading {current_dir}')
        for i, tr in enumerate(os.listdir(path + current_dir)):
            print(f'{current_dir}-{i}: {tr}')
            if((current_dir=="Positive" or current_dir=='Negative') and path=='/mnt/data/soroka_tomo/segmented_DBT_slices_soroka/Train/'):
                break
            try:
                img = torchvision.io.read_image(path=path + current_dir + "/" + tr)
            except:
                print('couldn\'t load '+tr)
            img = torchvision.transforms.functional.resize(img, size=[size, size])
            img = torch.permute(img, (0, 1, 2))
            normalized_img = img / 255
            img_path = path+current_dir+'/'+tr
            save_image(normalized_img, img_path)
            # labels.append(current_dir)
            # images.append(img)
            # if plot_example_labels and i == 0:
            #     print(f'{current_dir} Example Image:')
            #     plt.imshow(img.permute(1, 2, 0), cmap='gist_gray')
            #     plt.show()
            # collected = gc.collect()
            # print("Garbage collector: collected",
            #       "%d objects." % collected)
            # print(tr)
        print(f'finished {current_dir}')
#                except:
 #                   if 'desktop' not in tr:
  #                      print(f'Failed to load: {tr}')

    # return torch.stack(images)


# def convert_labels(labels: List) -> List:
#     return [torch.tensor([LABELS_ENCODING[label]]) for label in labels]









