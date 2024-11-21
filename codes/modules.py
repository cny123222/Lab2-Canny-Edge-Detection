import cv2
import numpy as np
import  matplotlib.pyplot as plt
from typing import Literal, Union, Tuple
from images import Image
from utils import convolution


class Module:

    def __init__(self):
        pass

    def __call__(self):
        pass


class Sequential(Module):

    def __init__(self, *args):
        super().__init__()
        assert all(isinstance(arg, Module) for arg in args)
        self.modules = args

    def __call__(
            self, 
            img: Image
        ) -> Image:
        for module in self.modules:
            img = module(img)
        return img
    

class GrayScale(Module):

    def __init__(
            self, 
            type: Literal['average', 'eye'] = 'average'
    ):
        self.type = type

    def __call__(
            self, 
            img: Image
    ) -> Image:
        if self.type == 'average':
            img.data = img.data.mean(axis=2).astype(np.uint8)
        elif self.type == 'eye':
            weights = [0.299, 0.587, 0.114]
            img.data = np.dot(img.data, weights).astype(np.uint8)
        img.type = 'gray'
        return img
    

class GaussFilter(Module):

    def __init__(
            self,
            type: Literal['opencv', 'scratch'] = 'opencv',
            kernel_size: Union[Tuple[int, int], int] = (3, 3),
            sigma: int = 0
    ):
        self.type = type
        self.ksize = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.sigma = sigma

    def __call__(
            self, 
            img: Image
    ) -> Image:
        if self.type == 'opencv':
            img.data = cv2.GaussianBlur(img.data, self.ksize, self.sigma)
        else:
            self._create_Gauss_filter()
            img.data = convolution(img.data, self.kernel).astype(np.uint8)
        return img
    
    def _create_Gauss_filter(self):
        assert self.ksize[0] == self.ksize[1] and self.ksize[0] > 0 and self.ksize[0] % 2 == 1
        ksize = self.ksize[0]
        if self.sigma <= 0:
            self.sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        
        k = ksize // 2
        x, y = np.meshgrid(np.arange(1, ksize + 1), np.arange(1, ksize + 1))
        x = np.pow(x - (k + 1), 2)
        y = np.pow(y - (k + 1), 2)
        kernel = np.exp(-(x + y) / (2 * self.sigma**2)) / (2 * np.pi * self.sigma**2)
        self.kernel = kernel / kernel.sum()


class CalcGrad(Module):

    def __init__(
            self,
            operator: Literal['Roberts', 'Sobel', 'Prewitt', 'Canny'] = 'Sobel'
    ):
        self.operator = operator

    def __call__(
            self, 
            img: Image
    ) -> Image:
        self._create_conv_kernel()
        grad_x = convolution(img.data, self.s_x)
        grad_y = convolution(img.data, self.s_y)
        img.data = np.sqrt(grad_x ** 2 + grad_y ** 2)
        img.grad_tan = grad_y / (grad_x + 1e-6)
        return img
            
    def _create_conv_kernel(self):
        if self.operator == 'Roberts':
            self.s_x = np.array([[-1, 0], 
                                 [0, 1]])
            self.s_y = np.array([[0, -1], 
                                 [1, 0]])
        elif self.operator == 'Sobel':
            self.s_x = np.array([[-1, 0, 1], 
                                 [-2, 0, 2], 
                                 [-1, 0, 1]])
            self.s_y = np.array([[1, 2, 1], 
                                 [0, 0, 0], 
                                 [-1, -2, -1]])
        elif self.operator == 'Prewitt':
            self.s_x = np.array([[-1, 0, 1], 
                                 [-1, 0, 1], 
                                 [-1, 0, 1]])
            self.s_y = np.array([[1, 1, 1], 
                                 [0, 0, 0], 
                                 [-1, -1, -1]])
        elif self.operator == 'Canny':
            self.s_x = np.array([[-1, 1], 
                                 [-1, 1]])
            self.s_y = np.array([[1, 1], 
                                 [-1, -1]])


class NMSuppression(Module):

    def __init__(self):
        pass

    def __call__(
            self,
            img: Image
    ) -> Image:
        assert img.grad_tan is not None
        img_h, img_w = img.data.shape
        new_data = np.zeros_like(img.data)
        for i in range(img_h):
            for j in range(img_w):
                new_data[i][j] = img.data[i][j] if self._is_local_max(img, (i, j)) else 0
        img.data = new_data
        return img

    def _is_local_max(
            self, 
            img: Image, 
            coor: Tuple[int, int]
    ) -> bool:
        x, y = coor
        img_h, img_w = img.data.shape
        grad_tan = img.grad_tan[x][y]

        com_points = []
        if 0 <= grad_tan < 1:
            if x != 0 and y != img_w - 1:
                com_points.append(grad_tan * img.data[x - 1][y + 1] + 
                                  (1 - grad_tan) * img.data[x][y + 1])
            if x != img_h - 1 and y != 0:
                com_points.append(grad_tan * img.data[x + 1][y - 1] +
                                  (1 - grad_tan) * img.data[x][y - 1])
        elif grad_tan >= 1:
            reci_tan = 1 / grad_tan
            if x != 0 and y != img_w - 1:
                com_points.append(reci_tan * img.data[x - 1][y + 1] + 
                                  (1 - reci_tan) * img.data[x - 1][y])
            if x != img_h - 1 and y != 0:
                com_points.append(reci_tan * img.data[x + 1][y - 1] + 
                                  (1 - reci_tan) * img.data[x + 1][y])
        elif grad_tan <= -1:
            reci_tan = -1 / grad_tan
            if x != 0 and y != 0:
                com_points.append(reci_tan * img.data[x - 1][y - 1] + 
                                  (1 - reci_tan) * img.data[x - 1][y])
            if x != img_h - 1 and y != img_w - 1:
                com_points.append(reci_tan * img.data[x + 1][y + 1] + 
                                  (1 - reci_tan) * img.data[x + 1][y])
        elif -1 < grad_tan < 0:
            abs_tan = -grad_tan
            if x != img_h - 1 and y != img_w - 1:
                com_points.append(abs_tan * img.data[x + 1][y + 1] + 
                                  (1 - abs_tan) * img.data[x][y + 1])
            if x != 0 and y != 0:
                com_points.append(abs_tan * img.data[x - 1][y - 1] + 
                                  (1 - abs_tan) * img.data[x][y - 1])
                
        return all(img.data[x][y] > com_grad for com_grad in com_points)


class DoubleThreshold(Module):

    def __init__(
            self, 
            th1: int, 
            th2: int
    ):
        assert th1 < th2
        self.th_low = th1
        self.th_high = th2

    def __call__(
            self,
            img: Image
    ) -> Image:
        img_h, img_w = img.data.shape
        edge_x, edge_y = np.where(img.data >= self.th_high)
        edges = list(zip(edge_x, edge_y))
        while len(edges) > 0:
            x, y = edges.pop()
            img.data[x][y] = 255
            for dir_x, dir_y in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1), (1, 1), (-1, -1)]:
                new_x = x + dir_x
                new_y = y + dir_y
                if not (0 <= new_x < img_h) or not (0 <= new_y < img_w):
                    continue
                if self.th_low <= img.data[new_x][new_y] < self.th_high:
                    img.data[new_x][new_y] = 255
                    edges.append((new_x, new_y))
        img.data = np.where(img.data == 255, img.data, 0).astype(np.uint8)
        return img
        
if __name__ == '__main__':
    mod = Sequential(
        GrayScale(type='average'), 
        GaussFilter(kernel_size=(3, 3), sigma=0.1, type='opencv'), 
        CalcGrad(operator='Sobel'),
        NMSuppression(), 
        DoubleThreshold(30, 100)
    )
    img = Image(img_path="images/1.jpg")
    img = mod(img)
    img.show()