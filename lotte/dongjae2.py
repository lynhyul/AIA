import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

rotXdeg = 90
rotYdeg = 90
rotZdeg = 90
f = 250
dist = 300

def onRotXChange(val):
    global rotXdeg
    rotXdeg = val
def onRotYChange(val):
    global rotYdeg
    rotYdeg = val
def onRotZChange(val):
    global rotZdeg
    rotZdeg = val
def onFchange(val):
    global f
    f=val
def onDistChange(val):
    global dist
    dist=val

original = cv2.imread(f'../../data/image/train/5/33.jpg')
original2 = cv2.imread(f'../../data/image/train/5/33.jpg')
# original = cv2.resize(original,(400,400))
# original = cv2.imread(f'../../data/image/test/1670.jpg')





# while True:
kernel = np.array([[0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]])

# 커널 적용 



modified = np.ndarray(shape=original.shape,dtype=original.dtype)


h , w = original.shape[:2]

print(h, w)



rotX = (25)*np.pi/180
rotY = 0
rotZ = 0

A1= np.matrix([[1, 0, -w/2],
            [0, 1, -h/2],
            [0, 0, 0   ],
            [0, 0, 1   ]])

RX = np.matrix([[1,           0,            0, 0],
                [0,np.cos(rotX),-np.sin(rotX), 0],
                [0,np.sin(rotX),np.cos(rotX) , 0],
                [0,           0,            0, 1]])

RY = np.matrix([[ np.cos(rotY), 0, np.sin(rotY), 0],
                [            0, 1,            0, 0],
                [ -np.sin(rotY), 0, np.cos(rotY), 0],
                [            0, 0,            0, 1]])

RZ = np.matrix([[ np.cos(rotZ), -np.sin(rotZ), 0, 0],
                [ np.sin(rotZ), np.cos(rotZ), 0, 0],
                [            0,            0, 1, 0],
                [            0,            0, 0, 1]])

R = RX * RY * RZ

T = np.matrix([[1,0,0,0],
            [0,1,0,0],
            [0,0,1,dist],
            [0,0,0,1]])

A2= np.matrix([[f, 0, w/2,0],
            [0, f, h/2,0],
            [0, 0,   1,0]])

H = A2 * (T * (R * A1))



cv2.warpPerspective(original, H, (w, h), modified, cv2.INTER_CUBIC)
modified = cv2.resize(modified,(300,560))
modified = modified[0:356, 0:256].copy()
modified = cv2.resize(modified,(256,256))
modified = modified[18:250, 30:256].copy()
# modified = cv2.copyMakeBorder(modified, 18, 19, 0, 0, cv2.BORDER_REPLICATE)

# modified = cv2.filter2D(modified, -1, kernel)

# modified = cv2.copyMakeBorder(modified, 36, 36, 21, 20, cv2.BORDER_REPLICATE)


cv2.imshow('modified',modified)
cv2.imshow('origin',original2)
# cv2.imshow('origin2',original2)
cv2.waitKey(0)
cv2.destroyAllWindows()     

# cv2.imwrite(f'../../data/image/train/{i}/{c}.jpg', modified)

        


        

        
            