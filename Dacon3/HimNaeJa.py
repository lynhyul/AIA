import cv2
import numpy as np

# 노이즈 완벽 제거!!!
for i in range(0,20000) :
    source = cv2.imread("../data/csv/Dacon3/dirty_mnist_2nd/%05d.png"%i, cv2.IMREAD_GRAYSCALE)
    image3 = cv2.dilate(source, kernel=np.ones((2, 2), np.uint8), iterations=1)
    blur = cv2.GaussianBlur(source, (5,5), 10)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # cv2.imshow("source", source);
    # cv2.imshow("blur", blur);
    # cv2.imshow("result", th);
    cv2.imwrite("../data/csv/Dacon3/train/%05d.png"%i,th)
    cv2.waitKey(-1)
