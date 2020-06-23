import cv2
import numpy as np



# 画像ファイル
src = "./tmp/lena.png"




# グレースケール写真を呼び出す関数
def src2gray(src):
    #グレースケールで(flags=0)imreadできる
    gry =cv2.imread(src,0)
    print(gry.dtype)
    gryimg=cv2.imwrite("./tmp1/gry.jpg",gry)
    return gryimg



# 指定範囲の色のみ抽出する関数
def color_Msk(src):
    # カラーで(flags=1)imread1
    img = cv2.imread(src,1)    
    # brg2hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 取得する色の範囲を指定する
    lower_color = np.array([20, 50, 50])
    upper_color = np.array([255, 255, 255])
    # 指定した色に基づいたマスク画像の生成(inRange()でhsvを色の範囲で指定)
    img_mask = cv2.inRange(hsv, lower_color, upper_color)
    # フレーム画像とマスク画像の共通の領域を抽出する。
    img_color = cv2.bitwise_and(img, img, mask=img_mask)
    color_Msk=cv2.imwrite("./tmp1/hsv.jpg", img_color)
    return color_Msk


"""
エッジ検出の関数（エラーを吐いたためいったん放置）
def edd2canny(src):
    # 白黒画像で画像を読み込み(gryですね)
    # img = cv2.imread(src,0)
    img = src2gray(src) #関数呼び出し
    #エッジ検出
    print(img.dtype)
    # canny_img = cv2.Canny(img, 50, 110)

    # cv2.imwrite("./tmp1/canny.jpg", canny_img)
"""



def feature(src):
    # カラーで読み込み
    img = cv2.imread(src,1)
    # ORB
    detector = cv2.ORB_create()
    # 特徴検出
    keypoints = detector.detect(img)
    # 画像への特徴点の書き込み
    img_orb = cv2.drawKeypoints(img, keypoints, None)
    feature_img =cv2.imwrite("./tmp1/orb.jpg", img_orb)
    return feature_img

# 関数呼び出し
# color_Msk(src)
# edd2canny(src)

src2gray(src)
feature(src)