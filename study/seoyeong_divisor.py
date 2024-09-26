import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

'''
# 실행할 때 마다 같은 결과를 출력하기 위해 설정
np.random.seed(3)
tf.random.set_seed(3)
# 데이터 불러오기
data_set= np.loadtxt('C:/data/deeplearning/dataset/ThoraricSurgery.csv', delimiter=",")
## 데이터 환자의 기록과 수술결과 모델을 설정하고 실행
x= data_set[:, 0:17] # 행 / 열
y= data_set[:,17]
#딥러닝 구조
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#딥러닝 실행
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x,y,epochs=100, batch_size=10)
'''

#최소 제곱법 실행하는 식
x= [2,4,6,8]
y=[81,93,91,97]

#x, y 평균값
mx = np.mean(x)
my = np.mean(y)

#기울기 공식의 분모
divisor = sum([(mx - i)**2 for i in x])
#기울기 공식의 분자
def top(x, mx, y, my):
    d= 0
    for i in range(len(x)):
        d += (x[i] - mx)* (y[i]- my)
    return d

devidend = top(x, mx, y, my)

print('분모', divisor )
print('분자', devidend )

#x, y절편 구하기
a= devidend / divisor
b= my - (mx*a)
#출력으로 확인
print('기울기 a=', a)
print('y 절편 b=', b)