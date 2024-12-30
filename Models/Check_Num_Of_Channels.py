import PIL.Image
import numpy as np


def check_image_properties(image_path):
    # Load the image
    img = PIL.Image.open(image_path)
    img_array = np.array(img)

    print(f"Image shape: {img_array.shape}")

    if len(img_array.shape) == 3:
        # Check if all channels are identical
        channels_identical = np.array_equal(img_array[:, :, 0], img_array[:, :, 1]) and \
                             np.array_equal(img_array[:, :, 1], img_array[:, :, 2])
        print(f"Channels identical: {channels_identical}")
    else:
        print("Single channel image")

    print(f"Data type: {img_array.dtype}")
    print(f"Value range: [{img_array.min()}, {img_array.max()}]")

image_path = '/mnt/data/tomosynthesis_data/labeled_data_224/Test/Negative/AA3345_L_CC_8.png'
check_image_properties(image_path)
