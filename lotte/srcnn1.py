import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,Dropout,Activation,LeakyReLU,UpSampling2D,Input,Dense,Reshape,Flatten,Conv2DTranspose,ReLU,concatenate,ZeroPadding2D,UpSampling2D
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

start = 18000
end = 24000
start1 = 750
end1 = 1000

x = np.load('../../data/npy/super128.npy',allow_pickle=True)
y = np.load('../../data/npy/super256.npy',allow_pickle=True)
x_pred = np.load('../../data/image/npy/srcnn2.npy',allow_pickle=True)

x_pred = x_pred[start:end]

# plt.imshow(x_pred[0]) # 삭제 할 이미지 띄우기
# plt.show()

print(x_pred.shape)


x_train = x[:4000]
y_train = y[:4000]

x_test = x[4000:4500]
y_test = y[4000:4500]
# x_train, x_test, y_train, y_test = train_test_split(x,y,train_size= 0.8)

x_pred2 = x_pred

print(x_pred2.shape)

# x_train = x_train / 127.5-1
# x_test = x_test/ 127.5-1
# y_train = y_train/127.5 -1
# y_test = y_test/127.5 -1
x_pred = x_pred/127.5 -1

initializer = tf.random_normal_initializer(0.,0.02)

inputs = Input(shape=(256,256,3))
layer = UpSampling2D((2,2))(inputs)

layer1 = Conv2D(filters=64,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer)
layer1 = LeakyReLU()(layer1)
layer1_ = layer1

layer2 = Conv2D(filters=128,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer1)
layer2_ = BatchNormalization()(layer2)
layer2 = LeakyReLU()(layer2_)

layer3 = Conv2D(filters=256,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer2)
layer3_ = BatchNormalization()(layer3)
layer3 = LeakyReLU()(layer3_)

layer4 = Conv2D(filters=512,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer3)
layer4_ = BatchNormalization()(layer4)
layer4 = LeakyReLU()(layer4_)

layer5 = Conv2D(filters=512,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer4)
layer5_ = BatchNormalization()(layer5)
layer5 = LeakyReLU()(layer5_)

layer6 = Conv2D(filters=512,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer5)
layer6_ = BatchNormalization()(layer6)
layer6 = LeakyReLU()(layer6_)

layer7 = Conv2D(filters=512,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer6)
layer7_ = BatchNormalization()(layer7)
layer7 = LeakyReLU()(layer7_)



layer8 = Conv2D(filters=512,kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=initializer)(layer7)
layer8_ = BatchNormalization()(layer8)
layer8 = LeakyReLU()(layer8_)



layer9 = Conv2DTranspose(filters=512,kernel_size=4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer8)
layer9 = BatchNormalization()(layer9)
layer9 = layer9+layer7_
layer9 = Dropout(0.5)(layer9)
layer9 = ReLU()(layer9)

layer10 = Conv2DTranspose(filters=512,kernel_size=4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer9)
layer10 = BatchNormalization()(layer10)
layer10 = concatenate([layer10,layer6_])
layer10 = Dropout(0.5)(layer10)
layer10 = ReLU()(layer10)

layer11 = Conv2DTranspose(filters=512,kernel_size=4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer10)
layer11 = BatchNormalization()(layer11)
layer11 = layer11+layer5_
layer11 = Dropout(0.5)(layer11)
layer11 = ReLU()(layer11)
                      
layer12 = Conv2DTranspose(filters=512,kernel_size=4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer11)
layer12 = BatchNormalization()(layer12)
layer12 = layer12+layer4_
layer12 = ReLU()(layer12)
                      
layer13 = Conv2DTranspose(filters=256,kernel_size=4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer12)
layer13 = BatchNormalization()(layer13)
layer13 = layer13+layer3_
layer13 = ReLU()(layer13)

layer14 = Conv2DTranspose(filters=128,kernel_size=4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer13)
layer14 = BatchNormalization()(layer14)
layer14 = layer14+layer2_
layer14 = ReLU()(layer14)

# layer15 = Conv2DTranspose(filters=64,kernel_size=4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False)(layer14)
# layer15 = BatchNormalization()(layer15)
# layer15 = layer15+layer1_
# layer15 = ReLU()(layer15)


outputs = Conv2DTranspose(3,4,strides=2,padding='same',kernel_initializer=initializer,activation='tanh')(layer14)

Generator = Model(inputs=inputs,outputs=outputs)
Generator.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
cp = ModelCheckpoint('../../data/modelcheckpoint/srcnn.h5',monitor='val_acc',save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_acc',patience= 5,verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss',patience= 3, factor=0.3,verbose=1)

Generator.compile(loss = 'mae', optimizer = 'adam', metrics = ['acc'])
# Generator.fit(x_train, y_train, epochs = 30, batch_size=32, validation_data = (x_test,y_test), callbacks=[cp,early_stopping,lr])

Generator.load_weights('../../data/modelcheckpoint/srcnn.h5')

output = Generator.predict(x_pred)

output = ((output+1)*127.5).astype('int')

# from matplotlib import pyplot as plt
# import random
# fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
#         plt.subplots(3, 5, figsize = (20, 7))

# # 이미지 다섯개를 무작위로 고른다
# random_images = random.sample(range(output.shape[0]), 5)

# # 원본(입력) 이미지를 맨 위에 그린다!!
# for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
#     ax.imshow(x_pred2[random_images[i]])
#     if i==0:
#         ax.set_ylabel('INPUT', size = 20)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])

# # 오토인코더가 출력한 이미지를 아래에 그린다
# for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
#     ax.imshow(output[random_images[i]])
#     if i==0:
#         ax.set_ylabel('OUTPUT', size = 20)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
# # 결과 보기
# plt.tight_layout()
# plt.show()

def BGR2RGB(img): 
    b = img[:, :, 0].copy() 
    g = img[:, :, 1].copy() 
    r = img[:, :, 2].copy()
 # RGB > BGR 
    img[:, :, 0] = r 
    img[:, :, 1] = g 
    img[:, :, 2] = b 
    return img




for j in range(start1,end1) :
    for i in range(24) :
        b = 48+i
        c = 24*(j-start1) + i
        predict = BGR2RGB(output[c])
        cv2.imwrite(f'../../data/image/train/{j}/{b}.jpg', predict)
        print(f"{j}번째 폴더에서 생성중")
print("완료")