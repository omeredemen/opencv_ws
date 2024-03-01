import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image

img_bgr = cv2.imread("New_Zealand_Coast.jpg", cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


mtrx1 = np.ones(img_rgb.shape) * 0.8
mtrx2 = np.ones(img_rgb.shape) * 1.2

img_rgb_darker   = np.uint8(cv2.multiply(np.float64(img_rgb), mtrx1))
img_rgb_brighter = np.uint8(np.clip(cv2.multiply(np.float64(img_rgb), mtrx2), 0, 255))

plt.figure(figsize=[18,5])
plt.subplot(131); plt.imshow(img_rgb_darker);  plt.title("Lower Contrast");
plt.subplot(132); plt.imshow(img_rgb);         plt.title("Original");
plt.subplot(133); plt.imshow(img_rgb_brighter);plt.title("Higher Contrast");
plt.show()