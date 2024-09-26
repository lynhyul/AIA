import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


original = cv2.imread(f'../../data/image/test/1670.jpg')
kernel = np.array([[0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]])

# 커널 적용 
original2 = cv2.filter2D(original, -1, kernel)
# cv2.imshow('modified',modified)
cv2.imshow('origin',original)
cv2.imshow('origin2',original2)
cv2.waitKey(0)
cv2.destroyAllWindows()     
# cv2.imwrite(f'../../data/image/train//{c}.jpg', modified)

        


        

        
            