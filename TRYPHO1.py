import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
img = cv2.imread('PHCCC.JPEG',0)
cv2.IMREAD_COLOR
cv2.IMREAD_UNCHANGED
cv2.imshow('PHCCC.JPEG',img)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()
cv2.waitKey(0)
