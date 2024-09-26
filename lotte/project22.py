import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB4
import datetime
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation
from tensorflow.python.keras.applications.efficientnet import EfficientNetB7


SEED = 66
IMAGE_SIZE = (224,224,3)
EPOCH = 50
OPTIMIZER =Adam(learning_rate= 1e-3)

#data load
x = np.load("C:/data/npy/last_x.npy",allow_pickle=True)
y = np.load("C:/data/npy/last_y.npy",allow_pickle=True)
x_pred = np.load('C:/data/npy/test3.npy',allow_pickle=True) # test 3 => 224,224,3

print(x.shape)
print(x_pred.shape)

x = preprocess_input(x) 
x_pred = preprocess_input(x_pred)   

idg = ImageDataGenerator(
    width_shift_range=(-1,1),   
    height_shift_range=(-1,1)) 
   
idg2 = ImageDataGenerator()



class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y


# y = np.argmax(y, axis=1)

x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.8, shuffle = True, random_state=SEED)
train_generator  =  MixupGenerator (x_train , y_train , batch_size = 32 , alpha = 0.2 , datagen = idg) ()
valid_generator = idg2.flow(x_valid,y_valid)
test_generator = x_pred

from keras.utils.generic_utils import get_custom_objects
from tensorflow.python.keras.activations import swish
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})

md = EfficientNetB4(input_shape = IMAGE_SIZE, weights = "imagenet", include_top = False)
for layer in md.layers:
    layer.trainable = True
x = md.output
x = GlobalAvgPool2D(name='global_avg')(md.output)
prediction = Dense(1000, activation='softmax')(x)
model = Model(inputs=md.input, outputs=prediction)

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
cp = ModelCheckpoint('../../data/modelcheckpoint/lotte_projcet31.h5',monitor='val_acc',save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_acc',patience= 12,verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss',patience= 6, factor=0.3,verbose=1)

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER,
                metrics=['acc'])

t1 = time()       
# history = model.fit_generator(train_generator,
#     validation_data=valid_generator, epochs=EPOCH, steps_per_epoch=len(x_train) / 32, 
#                     validation_steps=len(valid_generator),callbacks=[early_stopping,lr,cp])

t2 = time()

print("execution time: ", t2 - t1)
# predict
model.load_weights('../../data/modelcheckpoint/lotte_projcet31.h5')
result = model.predict(x_pred,verbose=True) # 72000,1000

# 여러 모델들로 tta를 모방해보자
np.save('../../data/image/npy/x_pred8.npy', arr = result) # test 3 => x_pred1 / test 4=> x_pred2

sub = pd.read_csv('../../data/image/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('../../data/image/answer31.csv',index=False)

# 