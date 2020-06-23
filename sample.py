import cv2
import numpy as np


# 画像を読み込む
img = cv2.imread('./tmp/maru.png')
print(img.shape)

print(img.dtype)
# グレースケールに変換
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二値化する(白黒反転させた)
th,img_otu = cv2.threshold(img_gray, 128, 255,cv2.THRESH_BINARY_INV)
print(th)

# 輪郭を抽出する
# image, contours, hierarchy = cv2.findContours(入力画像, 抽出モード, 近似手法)
contours, hierarchy = cv2.findContours(img_otu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.imwrite('./tmp1/outline_img.jpg',contours)