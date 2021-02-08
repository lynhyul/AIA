# 실습!
import numpy as np
import numpy as np
from sklearn.datasets import load_boston
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.metrics import r2_score

#x, y = load_iris(return_X_y=True)

dataset = load_boston()
x = dataset.data
y = dataset.target
#print(dataset.DESCR)    
#print(dataset.feature_names)        # sepal(꽃받침), petal(꽃잎)

## 원핫인코딩
#from tensorflow.keras.utils import to_categorical # 케라스 2.0버전
#from keras.utils.np_utils import to_categorical -> 케라스 1.0버전(구버전)
#y = to_categorical(y)
#1. 데이터 / 전처리


print(x.shape)  # 569,30

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, shuffle = True, random_state = 110)
x_train,x_val , y_train, y_val = train_test_split(x_train,y_train, train_size = 0.8, shuffle = True, random_state = 110)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# #2. 모델
def build_model(drop=0.5, optimizer = 'adam', node = 1, layer_num = 1) :
    input = Input(shape=(x_train.shape[1]), name = 'input')
    x = Dense(512/node, activation='relu', kernel_initializer='normal', name= 'hidden1') (input)
    x = Dropout(drop)(x)
    for i in range(layer_num) :
        x = Dense(256/node, activation='relu', kernel_initializer='normal',name= f'hidden2_{i}') (x)
        x = Dropout(drop)(x)
    x = Dense(32, activation='relu',  kernel_initializer='normal',name= 'hidden3') (x)
    x = Dropout(drop)(x)
    output = Dense(1, kernel_initializer='normal',name = 'output') (x)
    model = Model(inputs = input, outputs = output)
    model.compile(optimizer = optimizer, metrics=['mae'],
                    loss = 'mse')
    return model

def create_hyperparmeters() :
    # batches = [10, 20, 30, 40, 50]
    opitmizer = ['adam']
    dropout = [0.2]
    node = [1,2,4]
    layer_num = [2,5,6,7,8]
    return {"optimizer" : opitmizer,
            "drop": dropout, "node" : node, "layer_num" : layer_num}    


hyperparmeters = create_hyperparmeters()
# model2 = build_model()    타입오류



from tensorflow.keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
model2 = KerasRegressor(build_fn=build_model, verbose = 1, epochs=1000)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

mc = ModelCheckpoint('../data/modelcheckpoint/hyper_boston.h5',save_best_only=True, verbose=1)
es = EarlyStopping(monitor = 'val_loss',patience=30)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience=15, factor=0.5)

search = RandomizedSearchCV(model2,hyperparmeters,cv =3)

search.fit(x_train, y_train,verbose = 1, validation_data = (x_val, y_val),callbacks = [es,lr,mc])

# score = search.score(x_test,y_test)
# print("score : ", score)

# # print(search.best_estimator_)   # 전체 파라미터 중에서 가장 좋은거
# # <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001CA15E32C40>

# print(search.best_score_)   # 밑에 있는 .score랑은 결과가 다르게 나온다.

print(search.best_params_)  # 선택한 파라미터중에서 가장 좋은거

y_pred = search.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ",r2)

'''
{'optimizer': 'adam', 'node': 1, 'layer_num': 5, 'drop': 0.2}
4/4 [==============================] - 0s 0s/step
r2 :  0.8476348568073538
'''