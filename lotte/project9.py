import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tqdm import tqdm

#데이터 지정 및 전처리
x = np.load('../../data/npy/train_data_x9.npy',allow_pickle=True)
x_pred = np.load('../../data/npy/predict_data9.npy',allow_pickle=True)
y = np.load("../../data/npy/P_project_y5.npy",allow_pickle=True)
# y1 = np.zeros((len(y), len(y.unique())))
# for i, digit in enumerate(y):
#     y1[i, digit] = 1


x = preprocess_input(x)
x_pred = preprocess_input(x_pred)



idg = ImageDataGenerator(
    # rotation_range=10, acc 하락
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1), 
    rotation_range=45, 
    # shear_range=0.2)    # 현상유지
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

idg2 = ImageDataGenerator()

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

# y = np.argmax(y, axis=1)

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.8, shuffle = True, random_state=66)


train_generator = idg.flow(x_train,y_train,batch_size=64, seed = 2048)
# seed => random_state
valid_generator = idg2.flow(x_valid,y_valid)
test_generator = idg2.flow(x_pred, shuffle=False)

mc = ModelCheckpoint('../../data/modelcheckpoint/lotte_projcet16.h5',save_best_only=True, verbose=1)




from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, GaussianDropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.activations import swish
from keras import regularizers
from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'swish': Activation(swish)})

efficientnet = EfficientNetB4(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
efficientnet.trainable = True
a = efficientnet.output
a = Dropout(0.3) (a)
a = Dense(256, activation= 'swish') (a)
a = GaussianDropout(0.4) (a)
a = GlobalAveragePooling2D() (a)
a = Dense(1000, activation= 'softmax') (a)

model = Model(inputs = efficientnet.input, outputs = a)
efficientnet.summary()


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stopping = EarlyStopping(patience= 10)
lr = ReduceLROnPlateau(patience= 5, factor=0.9, min_lr=1e-5)

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3),
                metrics=['acc'])
learning_history = model.fit_generator(train_generator,epochs=80, steps_per_epoch= len(x_train) / 64,
    validation_data=valid_generator, callbacks=[early_stopping,lr,mc])

# predict
model.load_weights('../../data/modelcheckpoint/lotte_projcet16.h5')
result = model.predict(x_pred,verbose=True)

sub = pd.read_csv('../../data/image/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('../../data/image/answer16.csv',index=False)


# tta_steps = 15
# predictions = []

# for i in tqdm(range(tta_steps)):
# 	# generator 초기화
#     test_generator.reset()
#     preds = model.predict_generator(generator = test_generator, verbose = 1)
#     predictions.append(preds)

# # 평균을 통한 final prediction
# pred = np.mean(predictions, axis=0)


# sub = pd.read_csv('../../data/image/sample.csv')
# sub['prediction'] = np.argmax(pred,axis = 1)
# sub.to_csv('../../data/image/answer15.csv',index=False)


#71점

