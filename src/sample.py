import cv2
import numpy as np

img = np.zeros((512,512,3), np.uint8)

cv2.imwrite('./tmp/sample.jpg',img)
