import os
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

class Image():
    """
    图像类
    """

    def __init__(
            self,
            img_path: str
    ):
        self.data = cv2.imread(img_path)
        self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)
        self.type = 'RGB'  # 图像类型

        self.img_path = img_path
        self.grad_tan = None  # 保存梯度方向
        self.hist = None
    
    def show(self):
        """
        展示图像
        """
        if self.type == 'RGB':
            plt.imshow(self.data)
        elif self.type == 'gray':
            plt.imshow(self.data, cmap='gray')
        plt.axis('off')
        plt.show()

    def save(
            self, 
            save_folder: str,
            save_name: str, 
            compare: Optional[np.ndarray] = None,
            dpi: int = 300,
    ):
        """
        保存图像

        参数:
        save_folder: 保存文件夹(允许不存在)
        save_name: 保存图像文件名称
        compare: 用于对比的opencv图像结果
        dpi: 图像分辨率, 默认值300

        返回值: None
        """
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        save_pth = os.path.join(save_folder, save_name)
        if os.path.exists(save_pth):
            warnings.warn(f"File '{save_name}' already exists in '{save_folder}'. Existing file will be overwritten.", UserWarning)

        # 绘制比较图
        if compare is not None:
            ori_img = cv2.imread(self.img_path)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

            fig, ax = plt.subplots(1, 3, figsize=(18, 6))

            # 绘制原图
            ax[0].imshow(ori_img)
            ax[0].axis('off')

            # 绘制opencv结果
            ax[1].imshow(compare, cmap='gray')
            ax[1].axis('off')

            # 绘制个人实现结果
            ax[2].imshow(self.data, cmap='gray')
            ax[2].axis('off')

            plt.tight_layout()
            plt.savefig(save_pth, dpi=dpi)

        # 保存普通图像
        else:
            plt.imshow(self.data, cmap="gray")
            plt.axis("off")
            plt.savefig(save_pth, bbox_inches='tight', pad_inches=0, dpi=dpi)