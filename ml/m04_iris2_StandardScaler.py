#다중분류
# standard scaler

import numpy as np
from sklearn.datasets import load_iris
import tensorflow as tf

from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#x, y = load_iris(return_X_y=True)

dataset = load_iris()
x = dataset.data
y = dataset.target
#print(dataset.DESCR)    
#print(dataset.feature_names)        # sepal(꽃받침), petal(꽃잎)
print(x.shape)      # (150,4)
print(y.shape)      # (150,)

print(x[:5])
print(y)


print(y)
print(x.shape)  # (150,4)
print(y.shape)  # (150,3)



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, 
                                                    random_state=110)

scaler = StandardScaler()
scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2. modeling

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input

# model = Sequential()
# model.add(Dense(10, input_shape=(4,)))
# model.add(Dense(5))
# model.add(Dense(3, activation= 'softmax'))  #다중분류에서는 가지고싶은 결과 수 만큼 입력한다.

# model = SVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = RandomForestClassifier()
model = DecisionTreeClassifier()

#3. compile fit

# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience= 5, mode = 'auto')

# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train)

#4. evaluate , predict

result = model.score(x_test,y_test)
print("result : ",result)


y_predict = model.predict(x)
acc = accuracy_score(y,y_predict)
print('accuracy_score : ',acc)


#결과치 나오게 코딩할것 argmax

'''
Deep learning
loss :  [0.01415738184005022, 1.0]
y_predcit:  [[1.7606923e-17 1.7337496e-07 9.9999988e-01]
 [4.2821070e-05 9.9479985e-01 5.1573813e-03]
 [9.9820244e-01 1.7975861e-03 2.6805536e-17]
 [2.8611670e-04 9.9705952e-01 2.6543736e-03]
 [9.9985671e-01 1.4330167e-04 4.4705715e-19]
 [9.3903748e-04 9.9903381e-01 2.7180373e-05]]
y_MaxPredict:  [2 1 0 1 0 1]      => 2 = (0~2)세 번째 인덱스가 가장 큰값 / 1 = 두 번째 인덱스가 가장
                                      큰 값
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]


Machine Learning

1. LinearSVC

result :  0.9333333333333333
y_predcit:  [1 1 1 1 1 2]
y_MaxPredict:  5
[0 0 0 0 0 0]

2. SVC
result :  0.9666666666666667
accuracy_score :  0.3333333333333333

3. KNeighborsClassifier
result :  0.9666666666666667
accuracy_score :  0.3333333333333333

4. RandomForestClassifier
result :  0.9333333333333333
accuracy_score :  0.3333333333333333

5. DecisionTreeClassifier
result :  0.9666666666666667
accuracy_score :  0.3333333333333333

 '''