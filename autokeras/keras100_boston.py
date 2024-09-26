## 1번 과제 boston

 

import numpy as np

import tensorflow as tf

import autokeras as ak

from tensorflow.keras.datasets import boston_housing

 

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

 

# print(x_train.shape, x_test.shape)  # 404,13 / 102, 13

# print(y_train.shape)

from sklearn.model_selection import train_test_split

(x_train,x_val , y_train, y_val) = train_test_split(x_train,y_train, train_size = 0.8)

 

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)

x_val = scaler.transform(x_val)

 

 

# x_train = x_train.reshape(x_train.shape[0]*x_train.shape[1],1)

# x_test = x_test.reshape(x_test.shape[0]*x_test.shape[1],1)

# x_val = x_val.reshape(x_val.shape[0]*x_val.shape[1],1)

 

 

 

# model = ak.ImageClassifier(

#     max_trials=2,

#     overwrite = True,

#     loss = 'mse',00000000000000000000000000000

#     metrics=['acc']

#     )

 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

 

es = EarlyStopping(verbose = 1, patience=20)

lr = ReduceLROnPlateau(factor=0.3, patience=10)

# ck = ModelCheckpoint('../data/modelcheckpoint/autokeras.h5', save_best_only=True,monitor='val_loss',verbose=1)

 

# model.summary() -> 먹히지 않았다.

model = ak.StructuredDataRegressor(overwrite=True, max_trials=10, loss = 'mse', metrics = ['mae'])

 

 

model.fit(x_train,y_train,epochs=500, validation_data = (x_val,y_val),

          callbacks=[es,lr]) # validation_split default => 0.2

 

from tensorflow.keras.models import load_model 

from sklearn.metrics import r2_score

 

best_model = model.tuner.get_best_model()

y_pred = best_model.predict(x_test)

r2 = r2_score(y_test, y_pred)

print("r2 : ",r2)

 

# r2 :  0.8302864121916621

 

 

model2 = model.export_model()

try:

    model2.save('C:/data/modelsave/ak_save_boston', save_format='tf')

except:

    model2.save('C:/data/modelsave/ak_save_noston.h5')

 

model3 = load_model('C:/data/modelsave/ak_save_boston', custom_objects=ak.CUSTOM_OBJECTS)

result_boston = model3.evaluate(x_test, y_test)

 

y_pred = model3.predict(x_test)

r2 = r2_score(y_test, y_pred)

 

print("load_result :", result_boston, r2)

 

# load_result : [14.12761116027832, 2.7106502056121826] 0.8302864121916621