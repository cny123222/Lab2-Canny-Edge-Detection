import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional

class Image():

    def __init__(
            self,
            img_path: Optional[str] = None,
            img_data: Optional[np.ndarray] = None,
    ):
        if img_path is not None:
            self.data = cv2.imread(img_path)
            self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)
            self.type = 'RGB'
        else:
            self.data = img_data
            self.type = 'gray'

        self.grad_tan = None
    
    def show(self):
        if self.type == 'RGB':
            plt.imshow(self.data)
        elif self.type == 'gray':
            plt.imshow(self.data, cmap='gray')
        plt.show()

if __name__ == '__main__':
    img = Image("images/1.jpg")
    img.show()