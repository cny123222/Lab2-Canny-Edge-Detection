import cv2
import numpy as np
import  matplotlib.pyplot as plt
from typing import Literal, Union, Tuple, Optional
from images import Image
from utils import convolution


class Module:
    """
    模块类
    """

    def __init__(self):
        pass

    def __call__(self):
        pass


class Sequential(Module):
    """
    实现模块的顺序连接
    """

    def __init__(
            self, 
            *args
    ):
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
    """
    灰度化模块
    """

    def __init__(
            self, 
            type: Literal['average', 'eye'] = 'average'
    ):
        """
        参数:
        type: 灰度化类型 (平均灰度或人眼灰度)
        """
        super().__init__()
        self.type = type

    def __call__(
            self, 
            img: Image
    ) -> Image:
        # 平均灰度
        if self.type == 'average':
            img.data = img.data.mean(axis=2).astype(np.uint8)
        # 人眼灰度
        elif self.type == 'eye':
            weights = [0.299, 0.587, 0.114]
            img.data = np.dot(img.data, weights).astype(np.uint8)
        img.type = 'gray'
        return img
    

class GaussFilter(Module):
    """
    高斯滤波模块
    """

    def __init__(
            self,
            type: Literal['opencv', 'custom'] = 'opencv',
            kernel_size: Union[Tuple[int, int], int] = (3, 3),
            sigma: int = 0
    ):
        """
        参数:
        type: 实现类型 (OpenCV实现或个人实现)
        kernel_size: 高斯核尺寸
        sigma: 高斯核标准差 (sigma=0表示自动生成)
        """
        super().__init__()
        self.type = type
        self.ksize = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.sigma = sigma

    def __call__(
            self, 
            img: Image
    ) -> Image:
        # OpenCV实现
        if self.type == 'opencv':
            img.data = cv2.GaussianBlur(img.data, self.ksize, self.sigma)
        # 个人实现
        else:
            self._create_Gauss_filter()
            img.data = convolution(img.data, self.kernel).astype(np.uint8)
        return img
    
    def _create_Gauss_filter(self):
        """
        生成高斯卷积核
        """
        # 要求卷积核尺寸是正方形且长宽为奇数
        assert self.ksize[0] == self.ksize[1] and self.ksize[0] > 0 and self.ksize[0] % 2 == 1
        ksize = self.ksize[0]

        # 自动生成标准差
        if self.sigma <= 0:
            self.sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        
        # 计算和生成卷积核
        k = ksize // 2
        x, y = np.meshgrid(np.arange(1, ksize + 1), np.arange(1, ksize + 1))
        x = np.pow(x - (k + 1), 2)
        y = np.pow(y - (k + 1), 2)
        kernel = np.exp(-(x + y) / (2 * self.sigma**2)) / (2 * np.pi * self.sigma**2)
        self.kernel = kernel / kernel.sum()  # 归一化


class CalcGrad(Module):
    """
    梯度计算模块
    """

    def __init__(
            self,
            operator: Literal['Roberts', 'Sobel', 'Prewitt', 'Canny'] = 'Sobel'
    ):
        """
        参数:
        operator: 梯度算子类型
        """
        super().__init__()
        self.operator = operator

    def __call__(
            self, 
            img: Image
    ) -> Image:
        self._create_conv_kernel()
        grad_x = convolution(img.data, self.s_x)  # 计算x方向梯度
        grad_y = convolution(img.data, self.s_y)  # 计算y方向梯度
        img.data = np.sqrt(grad_x ** 2 + grad_y ** 2)  # 计算梯度幅值
        img.grad_tan = grad_y / (grad_x + 1e-6)  # 计算梯度方向
        return img
            
    def _create_conv_kernel(self):
        """
        生成梯度算子的卷积模板
        """
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
    """
    非极大值抑制模块
    """

    def __init__(self):
        super().__init__()

    def __call__(
            self,
            img: Image
    ) -> Image:
        assert img.grad_tan is not None
        img_h, img_w = img.data.shape
        new_data = np.zeros_like(img.data)
        for i in range(img_h):
            for j in range(img_w):
                # 将非局部最大值的像素值置零
                new_data[i][j] = img.data[i][j] if self._is_local_max(img, (i, j)) else 0
        img.data = new_data
        return img

    def _is_local_max(
            self, 
            img: Image, 
            coor: Tuple[int, int]
    ) -> bool:
        """
        判断某个点是否为局部最大值
        """
        x, y = coor
        img_h, img_w = img.data.shape
        grad_tan = img.grad_tan[x][y]

        com_points = []  # 保存需要比较的点的梯度幅值
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
    """
    双阈值检测及边缘连接模块
    """

    def __init__(
            self, 
            th_low: Optional[int] = None, 
            th_high: int = 150
    ):
        """
        参数:
        th_low: 低阈值, 默认值为0.4 * 高阈值
        th_high: 高阈值, 默认值150
        """
        # 设置默认低阈值
        if th_low is None:
            th_low = 0.4 * th_high

        assert th_low < th_high
        self.th_low = th_low
        self.th_high = th_high

    def __call__(
            self,
            img: Image
    ) -> Image:
        img_h, img_w = img.data.shape
        # 选出强边缘
        edge_x, edge_y = np.where(img.data >= self.th_high)
        edges = list(zip(edge_x, edge_y))  # 保存要处理的像素点, 初始为所有强边缘点
        while len(edges) > 0:
            x, y = edges.pop()
            img.data[x][y] = 255
            # 寻找周围是否有弱边缘点
            for dir_x, dir_y in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, -1), (-1, 1), (1, 1), (-1, -1)]:
                new_x = x + dir_x
                new_y = y + dir_y
                if not (0 <= new_x < img_h) or not (0 <= new_y < img_w):  # 越界保护
                    continue
                if self.th_low <= img.data[new_x][new_y] < self.th_high:
                    img.data[new_x][new_y] = 255
                    edges.append((new_x, new_y))
        img.data = np.where(img.data == 255, img.data, 0).astype(np.uint8)  # 将非边缘点全部置零
        return img