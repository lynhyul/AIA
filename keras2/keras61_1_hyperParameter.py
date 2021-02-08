import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test,y_test) = mnist.load_data()

#1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.


#2. 모델
def build_model(drop=0.5, optimizer = 'adam') :
    input = Input(shape=(28*28), name = 'input')
    x = Dense(512, activation='relu', name= 'hidden1') (input)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name= 'hidden2') (x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name= 'hidden3') (x)
    x = Dropout(drop)(x)
    output = Dense(10, activation='softmax', name = 'output') (x)
    model = Model(inputs = input, outputs = output)
    model.compile(optimizer = optimizer, metrics=['acc'],
                    loss = 'categorical_crossentropy')
    return model

def create_hyperparmeters() :
    batches = [10, 20, 30, 40, 50]
    opitmizer = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1,0.2,0.3]
    return {"batch_size" : batches, "optimizer" : opitmizer,
            "drop": dropout}    


hyperparmeters = create_hyperparmeters()
# model2 = build_model()    타입오류

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose = 1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = RandomizedSearchCV(model2,hyperparmeters,cv =3)

search.fit(x_train, y_train,verbose = 1)

print(search.best_params_)  # 선택한 파라미터중에서 가장 좋은거
# {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 50}
print(search.best_estimator_)   # 전체 요소 중에서 가장 좋은거
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001CA15E32C40>
print(search.best_score_)   # 밑에 있는 .score랑은 결과가 다르게 나온다.
# 0.9548166592915853

acc = search.score(x_test,y_test)
print("최종 스코어 : ",acc)

# 최종 스코어 :  0.9603999853134155
