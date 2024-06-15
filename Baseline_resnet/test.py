import torch
import torchvision
from torchvision.transforms.functional import get_image_size
from torchvision.utils import save_image

img = torchvision.io.read_image(path='pngs\Positive\AA0116_L_CC_10.png')
print(get_image_size(img))
print(img)
# img = torchvision.transforms.functional.resize(img, size=[224, 224])
# img = torch.permute(img, (0, 1, 2))
# normalized_img = img / 255
# save_image(normalized_img, 'pngs\Positive\AA0116_R_CC_0.png')
# print(get_image_size(img))
# print(img)