###### 2번 과제  iris

import numpy as np
import tensorflow as tf
import autokeras as ak
from sklearn.datasets import load_iris

dataset = load_iris()
x = dataset.data
y = dataset.target


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
one = OneHotEncoder()
y = y.reshape(-1,1)
one.fit(y)
y = one.transform(y).toarray()

#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)
print(y)
print(x.shape)  # (150,4)
print(y.shape)  # (150,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, 
                                                    random_state=110)
x_train, x_val, y_train, y_val = train_test_split(x,y,train_size = 0.8)


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
#     loss = 'mse',
#     metrics=['acc']
#     )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(verbose = 1, patience=10)
lr = ReduceLROnPlateau(factor=0.3, patience=10)
# ck = ModelCheckpoint('../data/modelcheckpoint/autokeras.h5', save_best_only=True,monitor='val_loss',verbose=1)

# model.summary() -> 먹히지 않았다.
model = ak.StructuredDataClassifier(overwrite=True, max_trials=3)


model.fit(x_train,y_train,epochs=100, validation_data = (x_val,y_val),
          callbacks=[es,lr]) # validation_split default => 0.2

from tensorflow.keras.models import load_model 
from sklearn.metrics import r2_score

model2 = model.export_model()
try:
    model2.save('ak_save_iris', save_format='tf')
except:
    model2.save('ak_save_iris.h5')

best_model = model.tuner.get_best_model()

print(best_model)
model3 = load_model('ak_save_iris', custom_objects=ak.CUSTOM_OBJECTS)
result_iris = model3.evaluate(x_test, y_test)

y_pred = model3.predict(x_test)
r2 = r2_score(y_test, y_pred)

print("load_result :", result_iris, r2)