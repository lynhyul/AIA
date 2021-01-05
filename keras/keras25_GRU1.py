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
from tensorflow.keras.layers import Dense, GRU

model = Sequential()

model.add(GRU(10, activation='relu', input_shape = (3,1))) # 여기서 LSTM이 명시 된 부분은 아웃풋을 의미.
                                                            # (4,3,1)은 input_shape으로 할 때 제일 앞 숫자 무시 -> = (3,1)
                                                            # LSTM구조를 쓰려면 3차원이 되어야한다. / RNN 대부분이 마찬가지
model.add(Dense(20))        # 앞에 Dense이 명시 된 부분 또한 아웃풋을 의미한다.
model.add(Dense(10))
model.add(Dense(1))

model.summary()
'''
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 10)                480
_________________________________________________________________
dense (Dense)                (None, 20)                220
_________________________________________________________________
dense_1 (Dense)              (None, 15)                315       
_________________________________________________________________
dense_2 (Dense)              (None, 10)                160
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 11
=================================================================
Total params: 1,186
Trainable params: 1,186
Non-trainable params: 0
_________________________________________________________________

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 5)                 140
_________________________________________________________________
dense (Dense)                (None, 20)                120
_________________________________________________________________
dense_1 (Dense)              (None, 15)                315
_________________________________________________________________
dense_2 (Dense)              (None, 10)                160
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 11
=================================================================
Total params: 746
Trainable params: 746
Non-trainable params: 0

'''


# 3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x,y,epochs=200, batch_size=1)



#평가, 예측
loss = model.evaluate(x,y)
print(loss)

x_pred = np.array([5,6,7])       # (3,)
x_pred = x_pred.reshape(1, 3, 1)    # 1행3열을 1개씩(LSTM구조) 잘라서 사용하겠다. => (1, 3, 1)

result = model.predict(x_pred)
print(result)


