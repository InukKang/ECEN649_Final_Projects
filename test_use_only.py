import numpy as np
import  matplotlib.pyplot as plt
from PIL import Image
import sys
from func.haar_feature import region_sum
import cv2


img = np.asarray(Image.open(f'datasets\\train\\face\\face00001.pgm').convert('L'))
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img_rgb[:5, :5, 0] += 20
img_rgb[12:14, 15:20, 2] += 50
print(img_rgb[:, :, 0])
print(img_rgb[:, :, 1])
print(img_rgb[:, :, 2])
plt.imshow(img_rgb)
plt.show()