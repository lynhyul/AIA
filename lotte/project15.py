
# 데이터 변경 (300 size, 노제너레이터)

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
IMAGE_SIZE = (300,300,3)
EPOCH = 35
OPTIMIZER =Adam(learning_rate= 1e-3)

#data load
x = np.load("C:/data/npy/P_project_x10.npy",allow_pickle=True)
y = np.load("C:/data/npy/P_project_y10.npy",allow_pickle=True)
x_pred = np.load('C:/data/npy/test4.npy',allow_pickle=True)


print(x_pred.shape)

x = preprocess_input(x) 
x_pred = preprocess_input(x_pred)   

idg = ImageDataGenerator(
    width_shift_range=(-1,1),   
    height_shift_range=(-1,1),  
    shear_range=0.2,
    zoom_range = 0.1) 
   
idg2 = ImageDataGenerator()

# y = np.argmax(y, axis=1)

x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.8, shuffle = True, random_state=SEED)

# train_generator = idg.flow(x_train,y_train,batch_size=26)
# valid_generator = idg2.flow(x_valid,y_valid)
test_generator = x_pred

from keras.utils.generic_utils import get_custom_objects
from tensorflow.python.keras.activations import swish
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})

md = EfficientNetB4(input_shape = IMAGE_SIZE, weights = "imagenet", include_top = False)
for layer in md.layers:
    layer.trainable = True
x = md.output
x = Dropout(0.3) (x)
x = Dense(128,activation = 'swish') (x)
x = tf.keras.layers.GaussianDropout(0.4) (x)
x = GlobalAvgPool2D(name='global_avg')(md.output)
prediction = Dense(1000, activation='softmax')(x)
model = Model(inputs=md.input, outputs=prediction)

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
cp = ModelCheckpoint('../../data/modelcheckpoint/lotte_projcet23.h5',monitor='val_acc',save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_acc',patience= 6,verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss',patience= 3, factor=0.4,verbose=1)

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER,
                metrics=['acc'])

t1 = time()       
# history = model.fit(x_train, y_train, validation_data=(x_valid,y_valid), epochs=EPOCH, batch_size = 24,
#                   callbacks=[early_stopping,lr,cp])

t2 = time()

print("execution time: ", t2 - t1)
# predict
model.load_weights('../../data/modelcheckpoint/lotte_projcet23.h5')
result = model.predict(x_pred,verbose=True)

sub = pd.read_csv('../../data/image/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('../../data/image/answer23.csv',index=False)
