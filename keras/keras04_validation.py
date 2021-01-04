# 네이밍 룰
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1,2,3,4,5])         # x라는 문제
y_train = np.array([1,2,3,4,5])         # 머신이 학습 할 데이터, y라는 답지

x_validation = np.array([6,7,8])
y_validation = np.array([6,7,8])       # 훈련데이터만으론 신뢰성이 떨어지기때문에 검증용 데이터를 추가한다.

x_test  = np.array([9,10,11])
y_test  = np.array([9,10,11])          # 훈련하지 않는 데이터, 일차함수로 따지면 연장선, 평가 데이터


#2. 모델구성
model = Sequential()
model.add(Dense(500, input_dim=1, activation='relu' ))
model.add(Dense(40, input_dim=1, activation='relu'))
model.add(Dense(30, input_dim=1, activation='relu' ))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
#model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
#model.compile(loss='mse', optimizer='adam',metrics=['mse'])
model.compile(loss='mse', optimizer='adam',metrics=['mae'])         # 2개 이상을 사용할 때엔 리스트를 사용해준다.

model.fit(x_train, y_train, epochs =100, batch_size =1,
          validation_data=(x_validation,y_validation)               # 검증용 데이터를 통해서 정확성을 높인다. (평가 데이터와는 달리 훈련데이터와 같이 훈련시킨다.)
)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)                 # 가장 중요한 지표
print("loss : ", loss)

#result = model.predict([9])
result = model.predict(x_test)
print("result : ", result)
print(y_test)

