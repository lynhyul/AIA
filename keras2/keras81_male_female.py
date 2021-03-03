# vgg16으로 만들어봥!!


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization, Activation, GlobalAveragePooling2D

from keras.utils import np_utils

import os, glob, numpy as np
from PIL import Image
from numpy import asarray
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation

import scipy.signal as signal
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


#이미지 불러오기, 원핫인코딩은 덤으로!

img_dir = '../data/image/sex/'
categories = ['0', '1'] # 남자를 1로 했어염, 이건 파일명으로 해주시면 됩니다.
np_classes = len(categories)

image_w = 255
image_h = 255


pixel = image_h * image_w * 3

X = []
y = []

for idx, cat in enumerate(categories):
    img_dir_detail = img_dir + "/" + cat
    files = glob.glob(img_dir_detail+"/*.jpg")


    for i, f in enumerate(files):
        try:
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            #Y는 0 아니면 1이니까 idx값으로 넣는다.
            X.append(data)
            y.append(idx)
            if i % 300 == 0:
                print(cat, " : ", f)
        except:
            print(cat, str(i)+" 번째에서 에러 ")
X = np.array(X)
Y = np.array(y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

xy = (X_train, X_test, Y_train, Y_test)
np.save("../data/npy/sex1.npy", xy)


# 예측할 이미지를 불러오기
img1=[]
for i in range(0,5):
    try :
        filepath='../data/image/my/me/%d.jpg'%i
        image2=Image.open(filepath)
        image2 = image2.convert('RGB')
        image2 = image2.resize((255,255))
        image_data2=asarray(image2)
        # image_data2 = signal.medfilt2d(np.array(image_data2), kernel_size=3)
        img1.append(image_data2)
    except :
        pass
np.save("../data/npy/sex_pred.npy",arr=img1)


# 데이터 로드
x_train, x_test, y_train, y_test = np.load("../data/npy/sex1.npy",allow_pickle=True)
x_pred = np.load("../data/npy/sex_pred.npy",allow_pickle=True)
print(y_train.shape)
print(x_train.shape)

# 전처리(vgg16.trainalbe = False로 할 경우 해주고, true할거면 안해줘도 됩니다. 안해주는게 더 잘 나오는경우도 있음)
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
x_pred = preprocess_input(x_pred)

#모델
vgg16 = VGG16(weights='imagenet',input_shape =(255,255,3),include_top=False)
vgg16.trainalbe = False
model = Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D()) # hidden레이어 이전에 사용. 공간 데이터에 대한 글로벌 평균 풀링 작업
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))   # BatchNormalization이후에 액티베이션 함수 사용
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation= 'sigmoid'))



# model2 = load_model('../data/modelcheckpoint/myproject.hdf5', compile=False)

#컴파일,훈련
model.compile(loss='binary_crossentropy', optimizer=Adam(1e-5), metrics=['acc'])
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
model_path = '../data/modelcheckpoint/croll.hdf5'
checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=50)
lr = ReduceLROnPlateau(patience=25, factor=0.5,verbose=1)

history = model.fit(x_train, y_train, batch_size=16, epochs=100, validation_data=(x_test, y_test),callbacks=[early_stopping,
checkpoint,lr])


# 결과물 출력
print("정확도 : %.4f" % (model.evaluate(x_test, y_test)[1]))
result = model.predict_generator(x_pred,verbose=True)
result[result < 0.5] =0
result[result > 0.5] =1
np.where(result < 0.5, '남자', '여자')
print("남자일 확률은",result*100,"%입니다.")


# print('acc : ', acc[-1])
# print('val_acc : ', val_acc[-1])
# # print('loss : ', loss[:-1])
# # print('val_acc : ', val_loss[:-1])

