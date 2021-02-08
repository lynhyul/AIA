# 실습!
import numpy as np
import numpy as np
from sklearn.datasets import load_diabetes
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.metrics import r2_score

#x, y = load_iris(return_X_y=True)

dataset = load_diabetes()
x = dataset.data
y = dataset.target
#print(dataset.DESCR)    
#print(dataset.feature_names)        # sepal(꽃받침), petal(꽃잎)

## 원핫인코딩
#from tensorflow.keras.utils import to_categorical # 케라스 2.0버전
#from keras.utils.np_utils import to_categorical -> 케라스 1.0버전(구버전)
#y = to_categorical(y)
#1. 데이터 / 전처리


print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, shuffle = True, random_state = 42)


# #2. 모델
def build_model(drop=0.5, optimizer = 'adam', node = 1) :
    input = Input(shape=(x_train.shape[1]), name = 'input')
    x = Dense(512/node, activation='relu', name= 'hidden1') (input)
    x = Dropout(drop)(x)
    x = Dense(256/node, activation='relu', name= 'hidden2') (x)
    x = Dropout(drop)(x)
    x = Dense(128/node, activation='relu', name= 'hidden3') (x)
    x = Dropout(drop)(x)
    x = Dense(64/node, activation='relu', name= 'hidden4') (x)
    x = Dropout(drop)(x)
    x = Dense(32/node, activation='relu', name= 'hidden5') (x)
    x = Dropout(drop)(x)
    output = Dense(1, activation='relu', name = 'output') (x)
    model = Model(inputs = input, outputs = output)
    model.compile(optimizer = optimizer, metrics=['mae'],
                    loss = 'mse')
    return model

def create_hyperparmeters() :
    # batches = [10, 20, 30, 40, 50]
    opitmizer = ['rmsprop', 'adam']
    dropout = [0.2]
    validation_split = [0.1,0.2,0.3]
    node = [1,2,4]
    return {"optimizer" : opitmizer,
            "drop": dropout, "validation_split" : validation_split, "node" : node}    


hyperparmeters = create_hyperparmeters()
# model2 = build_model()    타입오류



from tensorflow.keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
model2 = KerasRegressor(build_fn=build_model, verbose = 1, epochs=100)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

mc = ModelCheckpoint('../data/modelcheckpoint/hyper_iris.h5',save_best_only=True, verbose=1)
es = EarlyStopping(monitor = 'val_loss',patience=20)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience=10, factor=0.5)

search = RandomizedSearchCV(model2,hyperparmeters,cv =3)

search.fit(x_train, y_train,verbose = 1, callbacks = [es,lr,mc])

print(search.best_params_)  # 선택한 파라미터중에서 가장 좋은거

# print(search.best_estimator_)   # 전체 파라미터 중에서 가장 좋은거
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001CA15E32C40>

print(search.best_score_)   # 밑에 있는 .score랑은 결과가 다르게 나온다.


acc = search.score(x_test,y_test)
print("최종 스코어 : ",acc)

y_pred = search.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ",r2)

'''
{'validation_split': 0.1, 'optimizer': 'adam', 'drop': 0.2}
0.8988323410352071
4/4 [==============================] - 0s 964us/step - loss: 0.2368 - acc: 0.9474
최종 스코어 :  0.947368443012237
'''