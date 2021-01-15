#실습
#다:다 mlp
# 1. R2             : 0.5이하 / 음수는 안돼
# 2. layer          : 5개 이상
# 3. node           : 각 10개이상
# 4. batch_size     : 8 이하
# 5. epochs         : 30이상

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(100), range(301,401), range(1,101),  range(100),  range(301,401)])
y = np.array([range(711,811), range(1,101)])

print(x.shape)  # (5,100)
print(y.shape)  # (2,100 )
x_pred2 = np.array([100,402,101,100, 401])
x_pred2 = x_pred2.reshape(1, 5)

print("x_pred2.shape : ", x_pred2.shape)    # x_pred2.shape = (1,5)

x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2 ,shuffle = True,\
    random_state=66)


print(x_train.shape)    # (80,3)
print(y_train.shape)    # (80,3)

#2. 모델 구성

# from keras.layers import Dense 위와 똑같이 불러오나, keras1에선 위 보다 불러오는게 조금 느려진다.

model = Sequential()
model.add(Dense(10,input_dim =5))
model.add(Dense(15))
model.add(Dense(14))
model.add(Dense(13))
model.add(Dense(10))
model.add(Dense(2))


#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam', metrics=['mae']) 
model.fit(x_train,y_train, epochs = 30, batch_size=8, validation_split=0.4)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss :',loss)
print('mae :',mae)

y_predict = model.predict(x_test)
print(y_predict)


from sklearn.metrics import mean_squared_error      #mse
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))       #sqrt -> 루트를 씌우다
print("RMSE : ", RMSE (y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ",r2)

y_pred2 = model.predict(x_pred2)
print(y_pred2)


#RMSE :  21.83669712083112
#R2 :  0.3966534871342398
