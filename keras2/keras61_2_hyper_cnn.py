import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test,y_test) = mnist.load_data()

#1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.


#2. 모델
def build_model(drop=0.5, node = 2, activation = 'relu', lr = 0.001) :
    input = Input(shape=(28,28,1), name = 'input')
    x = Conv2D(filters = 128, kernel_size=(5,5), activation= activation, 
                padding= 'same', name= 'cnn1') (input)
    x = Dropout(drop) (x)
    x = Conv2D(filters = 64, kernel_size=(2,2), activation= activation, 
                padding= 'same', name= 'cnn2') (x)
    x = Dropout(drop) (x)       
    x = Conv2D(filters = 32, kernel_size=(2,2), activation= activation, 
                padding= 'same', name= 'cnn3') (x)
    x = Dropout(drop) (x)
    x = MaxPooling2D(2,2) (x)
    x = Conv2D(filters = 16, kernel_size=(3,3), activation= activation, 
                padding= 'same', name= 'cnn4') (x)
    x = Dropout(drop) (x)
    x = Conv2D(filters = 16, kernel_size=(3,3), activation= activation, 
                padding= 'same', name= 'cnn5') (x)
    x = Dropout(drop) (x)
    x = Flatten() (x)
    x = Dense(256/node, activation=activation, name= 'hidden2') (x)
    x = Dropout(drop)(x)
    x = Dense(128/node, activation=activation, name= 'hidden3') (x)
    x = Dropout(drop)(x)
    x = Dense(64/node, activation=activation, name= 'hidden4') (x)
    x = Dropout(drop)(x)
    output = Dense(10, activation='softmax', name = 'output') (x)

    model = Model(inputs = input, outputs = output)

    model.compile(optimizer = Adam(learning_rate=lr), metrics=['acc'],
                    loss = 'categorical_crossentropy')
    return model

def create_hyperparmeters() :
    # batches = [10, 20, 30, 40, 50]
    activation = ['prelu','relu']
    node = [1,2,4]
    lr = [0.001,0.017,0.002]
    return {"activation" : activation, "node" : node, "lr" : lr}   
            # "batch_size" : batches,  


hyperparmeters = create_hyperparmeters()
# model2 = build_model()    타입오류

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(patience=10)
# lr = ReduceLROnPlateau(patience=5, factor=0.5)
model2 = KerasClassifier(build_fn=build_model, verbose = 1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = RandomizedSearchCV(model2,hyperparmeters,cv =3)

search.fit(x_train, y_train,verbose = 1, batch_size=128, callbacks = [es])

print(search.best_params_)  # 선택한 파라미터중에서 가장 좋은거

# print(search.best_estimator_)   # 전체 파라미터 중에서 가장 좋은거
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001CA15E32C40>

print(search.best_score_)   # 밑에 있는 .score랑은 결과가 다르게 나온다.

acc = search.score(x_test,y_test)
print("최종 스코어 : ",acc)


'''
{'node': 1, 'lr': 0.002, 'activation': 'relu'}
0.9526333212852478
313/313 [==============================] - 1s 3ms/step - loss: 0.1446 - acc: 0.9644
최종 스코어 :  0.9643999934196472
'''
