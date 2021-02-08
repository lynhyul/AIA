# 61카피해서
# model.cv_result를 붙여서 완성


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
    # batches = [10, 20, 30, 40, 50]
    opitmizer = ['rmsprop', 'adam']
    dropout = [0.2]
    validation_split = [0.1,0.2,0.3]
    return {"optimizer" : opitmizer,
            "drop": dropout, "validation_split" : validation_split}    


hyperparmeters = create_hyperparmeters()
# model2 = build_model()    타입오류



from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose = 1, epochs=100)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

mc = ModelCheckpoint('../data/modelcheckpoint/hyper1.h5',save_best_only=True, verbose=1)
es = EarlyStopping(monitor = 'val_loss',patience=5)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience=3, factor=0.5)

search = RandomizedSearchCV(model2,hyperparmeters,cv =3)

search.fit(x_train, y_train,verbose = 1, callbacks = [es,lr,mc])


print(search.best_params_)  # 선택한 파라미터중에서 가장 좋은거
print(search.cv_results_)
# print(search.best_estimator_)   # 전체 파라미터 중에서 가장 좋은거
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001CA15E32C40>

print(search.best_score_)   # 밑에 있는 .score랑은 결과가 다르게 나온다.


acc = search.score(x_test,y_test)
print("최종 스코어 : ",acc)

'''
{'validation_split': 0.1, 'optimizer': 'adam', 'drop': 0.2}
{'mean_fit_time': array([16.80435642, 13.59764592, 12.5805498 , 20.29945127, 18.25182184,
       19.22486472]), 'std_fit_time': array([1.80190837, 0.84650948, 0.21553571, 2.97044106, 1.89012826,
       1.96257746]), 'mean_score_time': array([0.85641464, 0.85651747, 0.87768467, 0.83223645, 0.86074821,
       0.87816532]), 'std_score_time': array([0.06707903, 0.06244548, 0.06277007, 0.00857072, 0.05987186,
       0.05462596]), 'param_validation_split': masked_array(data=[0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
             mask=[False, False, False, False, False, False],
       fill_value='?',
            dtype=object), 'param_optimizer': masked_array(data=['rmsprop', 'rmsprop', 'rmsprop', 'adam', 'adam',
                   'adam'],
             mask=[False, False, False, False, False, False],
       fill_value='?',
            dtype=object), 'param_drop': masked_array(data=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
             mask=[False, False, False, False, False, False],
       fill_value='?',
            dtype=object), 'params': [{'validation_split': 0.1, 'optimizer': 'rmsprop', 'drop': 0.2}, {'validation_split': 0.2, 'optimizer': 'rmsprop', 'drop': 0.2}, {'validation_split': 0.3, 'optimizer': 'rmsprop', 'drop': 0.2}, {'validation_split': 0.1, 'optimizer': 
'adam', 'drop': 0.2}, {'validation_split': 0.2, 'optimizer': 'adam', 'drop': 0.2}, {'validation_split': 0.3, 'optimizer': 'adam', 'drop': 0.2}], 'split0_test_score': array([0.97570002, 0.97294998, 0.97755003, 0.98079997, 0.97780001,
       0.97955   ]), 'split1_test_score': array([0.97680002, 0.97485   , 0.97310001, 0.98110002, 0.97724998,
       0.97804999]), 'split2_test_score': array([0.97665   , 0.97334999, 0.97119999, 0.97885001, 0.97745001,
       0.97689998]), 'mean_test_score': array([0.97638335, 0.97371666, 0.97395001, 0.98025   , 0.9775    ,
       0.97816666]), 'std_test_score': array([0.00048705, 0.00081786, 0.00266116, 0.0009975 , 0.00022731,
       0.00108501]), 'rank_test_score': array([4, 6, 5, 1, 3, 2])}
0.9802500009536743
313/313 [==============================] - 0s 1ms/step - loss: 0.0731 - acc: 0.9842
최종 스코어 :  0.9842000007629395
'''