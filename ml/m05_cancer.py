

import numpy as np
from sklearn.datasets import load_breast_cancer
import tensorflow as tf

from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import accuracy_score,r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_diabetes

dataset = load_breast_cancer()

x= dataset.data
y = dataset.target

print(x[:5])
print(y[:10])
print(x.shape, y.shape) #(442, 10) (442,)





x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, 
                                                    random_state=110)
# x_train, x_val, y_train, y_val = train_test_split(x,y,train_size = 0.8)


# scaler = MinMaxScaler()
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


# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = RandomForestClassifier()
# model = DecisionTreeClassifier()
model = LogisticRegression()

#3. compile fit

# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience= 5, mode = 'auto')

# model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train)
# model.fit(x,y)

#4. evaluate , predict

result = model.score(x_test,y_test)
# result = model.score(x,y)
print("result : ",result)


y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
# r2 = r2_score(y_test,y_predict)
print('acc_score : ',acc)


'''
Deep learning (Tensorflow)
 [loss, accuracy, mae] :  [0.12791219353675842, 0.9649122953414917, 0.09085875749588013]


Machine Learning (train_test_split)

1. LinearSVC
result :  0.9473684210526315
acc_score :  0.9473684210526315

(2) MinMaxscaler
result :  0.9824561403508771
acc_score :  0.9824561403508771

(3) StandardScaler
result :  0.9736842105263158
acc_score :  0.9736842105263158


2. SVC
result :  0.8947368421052632
acc_score :  0.8947368421052632

(2) MinMaxscaler
result :  0.9824561403508771
acc_score :  0.9824561403508771

(3) StandardScaler
result :  0.9824561403508771
acc_score :  0.9824561403508771


3. KNeighborsClassifier
result :  0.9122807017543859
acc_score :  0.9122807017543859

(2) MinMaxscaler
result :  0.9912280701754386
acc_score :  0.9912280701754386

(3) StandardScaler
result :  0.9912280701754386
acc_score :  0.9912280701754386


4. RandomForestClassifier
result :  0.9298245614035088
acc_score :  0.9298245614035088

(2) MinMaxscaler
result :  0.9736842105263158
acc_score :  0.9736842105263158

(3) StandardScaler
result :  0.9736842105263158
acc_score :  0.9736842105263158


5. DecisionTreeClassifier
result :  0.956140350877193
acc_score :  0.956140350877193

(2) MinMaxscaler
result :  0.9473684210526315
acc_score :  0.9473684210526315

(3) StandardScaler
result :  0.9385964912280702
acc_score :  0.9385964912280702


6. LogisticRegression
result :  0.956140350877193
acc_score :  0.956140350877193

(2) MinMaxscaler
result :  0.9824561403508771
acc_score :  0.9824561403508771

(3) StandardScaler
result :  0.9824561403508771
acc_score :  0.9824561403508771



 '''