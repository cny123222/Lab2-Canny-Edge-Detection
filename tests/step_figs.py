from codes.modules import *

img = Image("images/2.jpg")

img = GrayScale()(img)
img.save("report_figures", "step1")

img = GaussFilter(kernel_size=(3, 3), sigma=0.2, type='opencv')(img)
img.save("report_figures", "step2")

img = CalcGrad(operator='Sobel')(img)
img.save("report_figures", "step3")

img = NMSuppression()(img)
img.save("report_figures", "step4")

img = DoubleThreshold(40, 120)(img)
img.save("report_figures", "step5")