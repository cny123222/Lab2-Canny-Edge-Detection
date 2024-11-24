import matplotlib.pyplot as plt
from main import edge_opencv

img = edge_opencv("images/3.jpg")
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.savefig("edges/3-OpenCV", bbox_inches='tight', pad_inches=0, dpi=300)