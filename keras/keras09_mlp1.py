# 다:1 mlp
import numpy as np
#x = np.arange([1,11)
x = np.array([[1,2,3,4,5,6,7,8,9,10], [11,12,13,14,15,16,17,18,19,20]])
y = np.arange(1,11)

#print(x.shape)  # (10,) -> 스칼라 10개
#print(x.shape)  # (2,10) -> 2행(스칼라10개)
#print(x.reshape(10,2))
''' # 결과값 (행과 열이 바뀐다.)
[[ 1  2]
 [ 3  4]
 [ 5  6]
 [ 7  8]
 [ 9 10]
 [ 1  2]
 [ 3  4]
 [ 5  6]
 [ 7  8]
 [ 9 10]]
'''
x=np.transpose(x)
print(x)
print(x.shape)  # (10, 2)
''' # 결과값
[[ 1 11]
 [ 2 12]
 [ 3 13]
 [ 4 14]
 [ 5 15]
 [ 6 16]
 [ 7 17]
 [ 8 18]
 [ 9 19]
 [10 20]]
 '''

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense 위와 똑같이 불러오나, keras1에선 위 보다 불러오는게 조금 느려진다.

model = Sequential()
model.add(Dense(10,input_dim =2))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam', metrics=['mae']) 
model.fit(x,y, epochs = 100, batch_size=1, validation_split=0.2)

#4. 평가, 예측
loss, mae = model.evaluate(x, y)
print('loss :',loss)
print('mae :',mae)

y_predict = model.predict(x)
print(y_predict)

'''
from sklearn.metrics import mean_squared_error      #mse
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))       #sqrt -> 루트를 씌우다
print("RMSE : ", RMSE (y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ",r2)
'''

'''
loss : 1.8469137330612284e-10
maae : 1.1456012543931138e-05
[[1.0000156]
 [2.0000103]
 [3.0000057]
 [4.000002 ]
 [4.9999976]
 [5.999994 ]
 [6.999989 ]
 [7.9999843]
 [8.999978 ]
 [9.999976 ]]
 '''