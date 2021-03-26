  
import numpy as np
import PIL
from numpy import asarray
from PIL import Image
import cv2


# 오픈 cv를 통해 전처리 후 128, 128로 리사이징 npy 저장!

img=[]
img_y=[]
for i in range(1000):
    for de in range(48,72):
        filepath='../../data/image/train/'+str(i)+'/'+str(de)+'.jpg'
        image=Image.open(filepath)
        #image_data = image.resize((128,128))
        # image = cv2.imread(filepath) # cv2.IMREAD_GRAYSCALE
        image = image.convert("RGB")
    # 커널 생성(대상이 있는 픽셀을 강조)
        kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

    # 커널 적용 
        # image_sharp = cv2.filter2D(image, -1, kernel)

        # image = cv2.resize(image, (256, 256))

        # cv2.imwrite(filepath, image)
        image_data = np.array(image)
        img.append(image_data)
        img_y.append(i)
    # cv2.imshow('origin',img[10])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  
np.save('../../data/image/npy/srcnn2.npy', arr=img)
# np.save('../../data/npy/train_data_y9.npy', arr=img_y)

# print("train 끝")

# img=[]
# for i in range(72000):
#     filepath='../../data/image/test/'+str(i)+'.jpg'
#     #image=Image.open(filepath)
#     #image_data = image.resize((128,128))
#     image = cv2.imread(filepath) # cv2.IMREAD_GRAYSCALE
#     kernel = np.array([[0, -1, 0],
#                     [-1, 5, -1],
#                     [0, -1, 0]])

#     # 커널 적용 
#     image_sharp = cv2.filter2D(image, -1, kernel)

#     image_sharp = cv2.resize(image_sharp, (224, 224))

#     image_data = np.array(image_sharp)
#     img.append(image_data)

# np.save('../../data/npy/test4.npy', arr=img)


# print("predict 끝")