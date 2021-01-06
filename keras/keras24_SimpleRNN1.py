#1. 데이터
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])

print("x.shape : " , x.shape)   # (4,3)
print("y.shape : " , y.shape)   # (4,)

x = x.reshape(4, 3, 1)          # [[[1],[2],[3]],[[4],[5],[6]......] // LSTM은 3차원 구조만을 받아들이기 때문에 리스트 성형
print("x.shape : " , x.shape)   # (4,3,1) => reshape 할 때, 곱셈의 결과는 같아야한다.

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

model = Sequential()

model.add(SimpleRNN(40, activation='relu', input_shape = (3,1)))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()


# 3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x,y,epochs=100, batch_size=1)



#평가, 예측
loss = model.evaluate(x,y)
print(loss)

x_pred = np.array([5,6,7])       # (3,)
x_pred = x_pred.reshape(1, 3, 1)    # 1행3열을 1개씩(LSTM구조) 잘라서 사용하겠다. => (1, 3, 1)

result = model.predict(x_pred)
print(result)


