


import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = '../data/csv/Dacon3/dirty_mnist_2nd/00000.png'

plt.figure(figsize=(12,6))
image = cv2.imread(image_path) # cv2.IMREAD_GRAYSCALE
img = cv2.imshow('original', image)
#cv2.waitKey(0)

#254보다 작고 0이아니면 0으로 만들어주기
image2 = np.where((image <= 254) & (image != 0), 0, image)
cv2.imshow('filterd', image2)

image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
cv2.imshow('dilate', image3)
#dilate -> 이미지 팽창
image4 = cv2.medianBlur(src=image3, ksize= 5)  #점처럼 놓여있는  noise들을 제거할수있음
cv2.imshow('median', image4)
#medianBlur->커널 내의 필터중 밝기를 줄세워서 중간에 있는 값으로 현재 픽셀 값을 대체

image5 = image4 - image2
cv2.imshow('sub', image5)

cv2.waitKey(0) #cv2 실행