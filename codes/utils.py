import cv2
import numpy as np

def convolution(img, kernel):
    """
    卷积函数 (默认填充方式为cv2.BORDER_DEFAULT)
    """
    img_h, img_w = img.shape
    ker_h, ker_w = kernel.shape

    # 填充边界
    pad_h = ker_h // 2
    pad_w = ker_w // 2
    img_pad = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_DEFAULT)

    # 卷积运算
    img_res = np.zeros_like(img, dtype=np.int64)
    for i in range(img_h):
        for j in range(img_w):
            img_res[i][j] = (img_pad[i:i + ker_h, j:j + ker_w] * kernel).sum()
    return img_res


if __name__ == '__main__':
    img = np.array([[1, 2, 3, 4, 5],
                    [4, 5, 6, 7, 8], 
                    [1, 2, 3, 4, 5], 
                    [2, 3, 4, 5, 6], 
                    [4, 5, 6, 7, 8]])
    kernel = np.array([[1, 2, 3],
                       [2, 1, 4], 
                       [3, 4, 100]])
    print(convolution(img, kernel))