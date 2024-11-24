from modules import *

img = Image("report_figures/grid.png")

img = GrayScale()(img)
img = GaussFilter(kernel_size=(3, 3), sigma=0.2, type='opencv')(img)

img = CalcGrad(operator='Canny')(img)
img.save("report_figures", "Canny")