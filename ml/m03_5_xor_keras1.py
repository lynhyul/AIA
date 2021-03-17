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
model.add(Dense(1, activation='relu', input_shape=(2,)))


#3. fit
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['acc'])
model.fit(x_data,y_data,batch_size=1, epochs=100)

#4. score, predict

predict = model.predict(x_data)
print(x_data,"의 예측결과 :",predict)

result = model.evaluate(x_data,y_data)
print("model.score : ",result[1])

'''
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 : [[0.00825479]
 [0.58028585]
 [0.20617221]
 [0.77820325]]

model.score :  0.5
'''

# acc = accuracy_score(y_data,predict)
# print('accuracy_score : ',acc)