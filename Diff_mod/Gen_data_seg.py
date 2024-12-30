import numpy as np
import os
import matplotlib.pyplot as plt
import math
import cv2
from PIL import Image
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
gen_files = os.listdir("/mnt/md0/royi/final_Project/fp_synthetic_images_5000/")
data_save_path = "/mnt/md0/royi/final_Project/seg_fp_synthetic_images_5000/"
def check_image(file):
    try:
        image = Image.open(file)
        image.verify()  # Verify the image integrity
        return True  # If no exception is raised, the image is valid
    except (OSError, ValueError) as e:
        # print(f"Image is truncated or corrupted: {file}, Error: {e}")
        return False
i = 0
for file in gen_files:
    if file is not "Gen_data_seg.py":
        is_valid = check_image(file)
        if is_valid:
            i += 1
            image = Image.open(file)
            image = np.array(image)  # Convert to NumPy array if needed
            im = image / 1023.
            rounded = np.zeros(im.shape)
            rounded[im >= 0.1] = 1
            result = np.where(rounded == 1.)
            segmented = image[np.min(result[0]):np.max(result[0]), np.min(result[1]):np.max(result[1])]
            # data_save_path = 'fp_synthetic_images_5000/seg_imgs/'
            filename = os.path.join(data_save_path, os.path.basename(file))
            print(filename)
            cv2.imwrite(filename, segmented)
            print(os.path.basename(file), '  -  Done!')
        else:
            print(f"{file} is invalid.")
            continue
print(i)