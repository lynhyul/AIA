import numpy as np
import PIL
from numpy import asarray
from PIL import Image

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

x = np.load('../data/csv/Dacon3/train.npy')
x_pred = np.load('../data/csv/Dacon3/test.npy') 

# (x_train,y_train),(x_valid,y_valid) = cifar10.load_data()

x = x[:500,:,:]

#노이즈 제거
threshold = 100
x[x < threshold] = 0
x[x > threshold] = 250
x_pred[x_pred < threshold] = 0
x_pred[x_pred > threshold] = 255
# x_train[x_train < threshold] = 0
# x_train[x_train > threshold] = 255

#전처리
# x = x.reshape(-1,256,256,1)
# x_pred = x_pred.reshape(-1,256,256,1)/255.


plt.figure(figsize=(20, 5))
ax = plt.subplot(2, 10, 1)
plt.imshow(x_pred[0])


ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()

# import cv2

# image_rgb = x_train[1]


# # 사각형 좌표: 시작점의 x,y  ,height, weight
# rectangle = (0, 0, 29, 29)

# # 초기 마스크 생성
# mask = np.zeros(image_rgb.shape[:2], np.uint8)

# # grabCut에 사용할 임시 배열 생성
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)

# # grabCut 실행
# cv2.grabCut(image_rgb, # 원본 이미지
#         mask,       # 마스크
#         rectangle,  # 사각형
#         bgdModel,   # 배경을 위한 임시 배열
#         fgdModel,   # 전경을 위한 임시 배열 
#         1,          # 반복 횟수
#         cv2.GC_INIT_WITH_RECT) # 사각형을 위한 초기화


# # 배경인 곳은 0, 그 외에는 1로 설정한 마스크 생성
# mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# # 이미지에 새로운 마스크를 곱행 배경을 제외
# image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
# # cv2.imwrite(f'../data/image/sex/sex1/male/image{i}.jpg',image_rgb_nobg)

# plt.imshow(image_rgb)
# plt.show()