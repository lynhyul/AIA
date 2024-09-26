# 나를 찍어서 내가 남자인지 여자인지 맞춰보자 ㅋㅋㅋ
#배경을 제거한 사진을 트레인데이터로 적용시켜봤다.


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import PIL
from numpy import asarray
from PIL import Image



train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip = True,
    # vertical_flip= True,
    width_shift_range=(-1,1),
    height_shift_range=(-1,1),
    # rotation_range= 5,
    # zoom_range= 0.5,
    # shear_range= 0.7,
    fill_mode = 'nearest',   #최근접기법, 패딩
    validation_split=0.2505
)

'''
- rotation_range: 이미지 회전 범위 (degrees)
- width_shift, height_shift: 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 
                                (원본 가로, 세로 길이에 대한 비율 값)
- rescale: 원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 
            모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 
            그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 
            이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.
- shear_range: 임의 전단 변환 (shearing transformation) 범위
- zoom_range: 임의 확대/축소 범위
- horizontal_flip`: True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다. 
    원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.
- fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
'''
test_datagen = ImageDataGenerator(rescale=1./255)   #test data는 따로 튜닝하지 않고 전처리만 해준다.

# flow or flow_from_directory


xy_train = train_datagen.flow_from_directory(
    '../data/image/sex2/sex1/',        
    target_size = (150,150),
    batch_size= 14,  
    class_mode='binary', 
    subset = 'training'

)

xy_test = train_datagen.flow_from_directory(
    '../data/image/sex2/sex1/',       
    target_size = (150,150),
    batch_size= 14,
    class_mode='binary', 
    subset = 'validation'
)

x_pred = train_datagen.flow_from_directory(
    '../data/image/my/',
    target_size = (150,150),
    batch_size= 14,
    class_mode='binary', 
)

# x_pred=[]
# for i in range(0,5):
#     filepath= f'../data/image/my/me/{i}.jpg'
#     image=Image.open(filepath)
#     image_data=asarray(image)
#     x_pred.append(image_data)



print(xy_train[0][0].shape) # (14, 150, 150, 3)


# print(xy_train)
# # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000002080FFD75E0>

# print(xy_train[0])
# # print(xy_train[0][0].shape) # (10, 150, 150, 3) = x
# # print(xy_train[15][1].shape) # (10,) = y
# # print(xy_train[15][1]) # [1. 1. 0. 1. 1. 1. 1. 1. 0. 0.]

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape =(150,150,3), activation= 'relu'))
model.add(Conv2D(filters = 16, kernel_size=(2,2), activation= 'relu'))
model.add(Conv2D(filters = 16, kernel_size=(2,2),  activation= 'relu'))
model.add(Conv2D(filters = 32, kernel_size=(2,2),  activation= 'relu'))
model.add(Conv2D(filters = 32, kernel_size=(3,3),  activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters = 64, kernel_size=(2,2),  activation= 'relu'))
model.add(Conv2D(filters = 64, kernel_size=(3,3),  activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
cp = ModelCheckpoint('../data/modelcheckpoint/myproject.hdf5',save_best_only=True) 
es = EarlyStopping(monitor= 'val_loss', patience=50)
lr = ReduceLROnPlateau(monitor='val_loss', patience=25, factor=0.5)
model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics=['acc'])
history = model.fit_generator(xy_train, steps_per_epoch=93, epochs=500, validation_data=xy_test, validation_steps=31,
callbacks=[es,lr,cp])
model2 = load_model('../data/modelcheckpoint/myproject.hdf5', compile=False)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss'] 
val_loss = history.history['val_loss']


print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])
# print('loss : ', loss[:-1])
# print('val_acc : ', val_loss[:-1])

result = model2.predict_generator(x_pred,verbose=True)
# result[result < 0.5] =0
# result[result > 0.5] =1
# np.where(result < 0.5, '남자', '여자')
print("남자일 확률은",result*100,"%입니다.")
