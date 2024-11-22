import cv2
import numpy as np
from modules import *

img1 = Image(img_path="images/2.jpg")
gs = GrayScale(type='average')
img_can = gs(img1)
img_can = cv2.Canny(img_can.data, 50, 150)  

img2 = Image(img_path="images/2.jpg")
mod = Sequential(
        GrayScale(type='average'), 
        GaussFilter(kernel_size=(3, 3), sigma=0.1, type='opencv'), 
        CalcGrad(operator='Sobel'),
        NMSuppression(), 
        DoubleThreshold(50, 150)
)
img_md = mod(img2)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(img_can, cmap='gray')
ax[0].axis('off')
ax[0].set_title('OpenCV-Canny')

ax[1].imshow(img_md.data, cmap='gray')
ax[1].axis('off')
ax[1].set_title('My-Canny')

plt.tight_layout()
plt.show()