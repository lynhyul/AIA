from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

#2. model

# model = LinearSVC()
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(2,)))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

#3. fit
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['acc'])
model.fit(x_data,y_data,batch_size=1, epochs=100)

#4. score, predict

predict = model.predict(x_data)
print(x_data,"의 예측결과 :",predict)

result = model.evaluate(x_data,y_data)
print("model.score : ",result[1])

'''
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 : [[0.21970147]
 [0.8589506 ]
 [0.9619319 ]
 [0.20681621]]
1/1 [==============================] - 0s 970us/step - loss: 0.1677 - acc: 1.0000
model.score :  1.0
'''

# acc = accuracy_score(y_data,predict)
# print('accuracy_score : ',acc)