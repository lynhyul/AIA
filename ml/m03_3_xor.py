from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score

#1. data
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

#2. model

model = LinearSVC()

#3. fit
model.fit(x_data,y_data)

#4. score, predict

predict = model.predict(x_data)
print(x_data,"의 예측결과 :",predict)

result = model.score(x_data,y_data)
print("model.score : ",result)


acc = accuracy_score(y_data,predict)
print('accuracy_score : ',acc)