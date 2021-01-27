

import numpy as np
from sklearn.datasets import load_wine
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

dataset = load_wine()

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
# model = LogisticRegression()
models = [LinearSVC(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
for i in models:
    model = i

    #3. compile fit
    model.fit(x_train,y_train)

    #4. evaluation, prediction
    print(f'\n{i}')
    result = model.score(x_test,y_test)
    print('model_score : ', result)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred,y_test)
    print('accuracy_score : ', accuracy)



'''
Machine Learning (train_test_split)

1. LinearSVC
result :  0.9166666666666666
acc_score :  0.9166666666666666

(2) MinMaxscaler
result :  0.9722222222222222
acc_score :  0.9722222222222222

(3) StandardScaler
result :  0.9722222222222222
acc_score :  0.9722222222222222


2. SVC
result :  0.6944444444444444
acc_score :  0.6944444444444444

(2) MinMaxscaler
result :  0.9722222222222222
acc_score :  0.9722222222222222

(3) StandardScaler
result :  0.9722222222222222
acc_score :  0.9722222222222222


3. KNeighborsClassifier
result :  0.6944444444444444
acc_score :  0.6944444444444444

(2) MinMaxscaler
result :  0.9166666666666666
acc_score :  0.9166666666666666

(3) StandardScaler
result :  0.9166666666666666
acc_score :  0.9166666666666666


4. RandomForestClassifier
result :  0.9444444444444444
acc_score :  0.9444444444444444

(2) MinMaxscaler
result :  0.9166666666666666
acc_score :  0.9166666666666666

(3) StandardScaler
result :  0.9444444444444444
acc_score :  0.9444444444444444


5. DecisionTreeClassifier
result :  0.8611111111111112
acc_score :  0.8611111111111112

(2) MinMaxscaler
result :  0.8333333333333334
acc_score :  0.8333333333333334

(3) StandardScaler
result :  0.8611111111111112
acc_score :  0.8611111111111112


6. LogisticRegression
result :  0.9444444444444444
acc_score :  0.9444444444444444

(2) MinMaxscaler
result :  0.9444444444444444
acc_score :  0.9444444444444444

(3) StandardScaler
result :  0.9166666666666666
acc_score :  0.9166666666666666



 '''