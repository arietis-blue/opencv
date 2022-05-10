from copy import copy
from http import server
import cv2
import numpy as np

# 画像の取得
img = cv2.imread("opencv_test/sample2.jpeg")
temp = cv2.imread("opencv_test/template2.jpeg")

# グレー変換
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

# テンプレート画像の高さ・幅
h, w = temp.shape
ser=img.copy()

# テンプレートマッチング
match = cv2.matchTemplate(gray, temp, cv2.TM_CCOEFF_NORMED)
threshold=0.9
loc=np.where(match>=threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(ser, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

#  結果の表示
cv2.imshow("searched",ser)
cv2.waitKey(0)
cv2.destroyAllWindows()