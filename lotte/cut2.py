import cv2
import numpy as np


# x = np.load("../data/npy/P_project_x5.npy",allow_pickle=True)
# src = cv2.imread('../data/image/test/0.jfif', cv2.IMREAD_COLOR)
# dst = src[0:300, 0:500].copy()
# dst = cv2.resize(dst,(224,224))

# width, height, channel = dst.shape
# print(width, height, channel)

# # cv2.imshow("src", src)
# cv2.imshow("dst", dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

for i in range(0,1000) :
    print(f'{i}번째 이미지폴더에서 생성중')
    # for j in range(24,48) :
    #     c = 95-j
    img = cv2.imread(f'../../data/image/train/{i}/30.jpg', cv2.IMREAD_COLOR)
    img = cv2.resize(img,(200,400))

    # img = cv2.resize(img,(256,256))
    img2 = img[0:250, 0:256].copy()
    img2 = cv2.resize(img2,(256,256))

    # kernel = np.array([[0, -1, 0],
    #                 [-1, 5, -1],
    #                 [0, -1, 0]])

    # # 커널 적용 
    # img2 = cv2.filter2D(img2, -1, kernel)

    # cv2.imshow("origin", img)
    # cv2.imshow('origin', original)
    # cv2.imshow('cut', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()     

    # cv2.imshow("origin", img)
    # cv2.imshow('origin', original)
    # cv2.imshow('cut', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()     
    cv2.imwrite(f'../../data/image/train/{i}/72.jpg', img2);
