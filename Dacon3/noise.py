import numpy as np
import PIL
from numpy import asarray
from PIL import Image

import matplotlib.pyplot as plt


x = np.load('../data/csv/Dacon3/train.npy')
x_pred = np.load('../data/csv/Dacon3/test.npy') 


x = x[:20000,:,:]

#노이즈 제거
threshold = 255
x[x < threshold] = 0
x[x > threshold] = 255
x_pred[x_pred < threshold] = 0
x_pred[x_pred > threshold] = 255

#전처리
x = x.reshape(-1,256,256,1)
x_pred = x_pred.reshape(-1,256,256,1)/255.


plt.figure(figsize=(20, 5))
ax = plt.subplot(2, 10, 1)
plt.imshow(x_pred[0])


ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
