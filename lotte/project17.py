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
EPOCH = 100
OPTIMIZER =Adam(learning_rate= 1e-3)

#data load
x = np.load("C:/data/npy/P_project_x12.npy",allow_pickle=True)
y = np.load("C:/data/npy/P_project_y12.npy",allow_pickle=True)
x_pred = np.load('C:/data/npy/test3.npy',allow_pickle=True) # test 3 => 224,224,3

print(x.shape)
print(x_pred.shape)

x = preprocess_input(x) 
x_pred = preprocess_input(x_pred)   

idg = ImageDataGenerator(
    width_shift_range=(-1,1),   
    height_shift_range=(-1,1),  
    shear_range=0.2) 
   
idg2 = ImageDataGenerator()

# y = np.argmax(y, axis=1)

x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.8, shuffle = True, random_state=SEED)

train_generator = idg.flow(x_train,y_train,batch_size=32)
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
x = Dropout(0.4) (x)
x = Dense(1024,activation = 'swish') (x)
x = tf.keras.layers.GaussianDropout(0.4) (x)
x = GlobalAvgPool2D(name='global_avg')(md.output)
prediction = Dense(1000, activation='softmax')(x)
model = Model(inputs=md.input, outputs=prediction)

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
cp = ModelCheckpoint('../../data/modelcheckpoint/lotte_projcet25.h5',monitor='val_acc',save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_acc',patience= 15,verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss',patience= 8, factor=0.2,verbose=1)

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER,
                metrics=['acc'])

t1 = time()       
# history = model.fit_generator(train_generator,
#     validation_data=valid_generator, epochs=EPOCH, steps_per_epoch=len(train_generator), 
#                     validation_steps=len(valid_generator),callbacks=[early_stopping,lr,cp])

t2 = time()

print("execution time: ", t2 - t1)
# predict
model.load_weights('../../data/modelcheckpoint/lotte_projcet25.h5')
result = model.predict(x_pred,verbose=True) # 72000,1000



sub = pd.read_csv('../../data/image/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('../../data/image/answer25.csv',index=False)

# 