import cv2
import numpy as np
from images import Image

def convolution(
        img: np.ndarray, 
        kernel: np.ndarray, 
        padding: int = cv2.BORDER_DEFAULT
):
    """
    卷积函数

    参数:
    img: 灰度图像
    kernel: 卷积核
    padding: 填充方式, 默认为BORDER_DEFAULT (反射填充)

    返回值: 卷积后图像
    """
    img_h, img_w = img.shape
    ker_h, ker_w = kernel.shape

    # 填充边界
    pad_h = ker_h // 2
    pad_w = ker_w // 2
    img_pad = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_DEFAULT)

    # 卷积运算
    new_img = np.zeros_like(img, dtype=np.int64)  # 卷积后数值会可能溢出
    for i in range(img_h):
        for j in range(img_w):
            new_img[i][j] = (img_pad[i:i + ker_h, j:j + ker_w] * kernel).sum()
    return new_img


def threshold_otsu(
        img: Image
):
    """
    Otsu自适应阈值选取

    Reference: https://github.com/khushitejwani/Canny-Edge-Detection-Using-Otsu-Thresholding
    """
    lower = np.floor(img.data.min())
    upper = np.ceil(img.data.max())
    hist, bin_edges = np.histogram(img.data, bins=np.arange(lower, upper + 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold