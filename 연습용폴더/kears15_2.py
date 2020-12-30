# 다:1 앙상블

import numpy as np

x1 = np.array([range(100), range(301,401), range(1,101)])
x2 = np.array([range(101,201),range(411,511), range(100,200)])

y1 = np.array([range(711,811), range(1,101), range(201,301)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.transpose(y1)

from sklearn.model_selection import train_test_split 

x1_train,x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1,x2,y1, test_size =0.2, shuffle = True,
                                                            random_state = 66)

#모델 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#입력1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu') (input1)
dense1 = Dense(15) (dense1)

#입력2
input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu') (input2)
dense2 = Dense(15) (dense2)
dense2 = Dense(30) (dense2)

#모델 병합
from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1,dense2])
middle1 = Dense(30,activation='relu') (merge1)
middle1 = Dense(60) (middle1)
middle1 = Dense(80) (middle1)
middle1 = Dense(40) (middle1)

#모델 출력

output = Dense(70) (middle1)
output = Dense(20) (output)
output = Dense(3) (output)

#모델 선언

model = Model(inputs = [input1,input2], outputs= output)

model.summary()


#컴파일 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train,x2_train],y1_train, batch_size=3, epochs = 200, validation_split=0.2 )

#예측 및 평가

loss = model.evaluate([x1_test,x2_test],y1_test, batch_size=1)
print("loss : ", loss)

y_predict = model.predict([x1_test,x2_test])
print(y_predict)

#RMSE, R2

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ",RMSE(y1_test,y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y_predict)
print("R2 : ",r2)
