import numpy as np

#1. 데이터
x1 = np.array([range(100), range(301,401), range(1,101)])
y1 = np.array([range(711,811), range(1,101), range(201,301)])

x2 = np.array([range(101,201),range(411,511), range(100,200)])
y2 = np.array([range(501,601),range(711,811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1,test_size= 0.2, shuffle = True, random_state = 66)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2,test_size= 0.2, shuffle = True, random_state = 66)

#모델 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#모델 1
input1 = Input(shape=(3,))
dense1 = Dense(100, activation='relu', name='input1_1') (input1)
dense1 = Dense(60, activation='relu', name='input1_2') (dense1)
dense1 = Dense(60, activation='relu', name='input1_3') (dense1)

#모델 2
input2 = Input(shape=(3,))
dense2 = Dense(100, activation='relu', name='input2_1') (input2)
dense2 = Dense(60, activation='relu', name='input2_2') (dense2)


#concatenate
from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(60, activation='relu', name='middle1') (merge1)
middle1 = Dense(100, activation='relu', name='middle2') (middle1)
middle1 = Dense(60, activation='relu', name='middle3') (middle1)
middle1 = Dense(80, activation='relu', name='middle4') (middle1)
middle1 = Dense(60, activation='relu', name='middle5') (middle1)

#모델 분기1
output1 = Dense(300, activation='relu', name='output1_1') (middle1)
output1 = Dense(600, activation='relu', name='output1_2') (output1)
output1 = Dense(400, activation='relu', name='output1_3') (output1)
output1 = Dense(3, name='Output1_4') (output1)

#모델 분기2
output2 = Dense(500, activation='relu', name='output2_1') (middle1)
output2 = Dense(100, activation='relu', name='output2_2') (output2)
output2 = Dense(200, activation='relu', name='output2_3') (output2)
output2 = Dense(3,name='Output2_4') (output2)

model = Model(inputs =[input1,input2], outputs = [output1,output2])

model.summary()


#컴파일 , 훈련

model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse'])
model.fit([x1_train,x2_train],[y1_train,y2_train], epochs=300, batch_size=3, validation_split=0.2)

# 예측 및 평가

loss = model.evaluate([x1_test,x2_test],[y1_test,y2_test], batch_size=1)
print("loss : ", loss)

y_predict, y2_predict = model.predict([x1_test,x2_test])
print("=================\n")
print(y_predict)
print("=================")
print(y2_predict)

#RMSE, R2

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict) :
    return np.sqrt(mean_squared_error(y_test ,y_predict))
RMSE1 = RMSE(y1_test, y_predict)
RMSE2 = RMSE(y2_test, y2_predict)    
print("RMSE : ",(RMSE1+RMSE2)/2)

from sklearn.metrics import r2_score
def R2(y_test,y_predict) :
    return r2_score(y_test,y_predict)
R2_1 = R2(y1_test,y_predict)
R2_2 = R2(y2_test,y2_predict)
print("R2 : ", (R2_1+R2_2)/2)

