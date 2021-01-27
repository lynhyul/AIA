

import numpy as np
from sklearn.datasets import load_boston
import tensorflow as tf

from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import accuracy_score,r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_diabetes

dataset = load_boston()

x= dataset.data
y = dataset.target

print(x[:5])
print(y[:10])
print(x.shape, y.shape) #(442, 10) (442,)





x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, 
                                                    random_state=110)
# x_train, x_val, y_train, y_val = train_test_split(x,y,train_size = 0.8)


# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


#2. modeling
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = RandomForestClassifier()
# model = DecisionTreeClassifier()
# model = LogisticRegression()
models = [KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LinearRegression()]
for i in models:
    model = i
# model = LinearRegression()
    #3. compi
    model.fit(x_train,y_train)
    print(f'\n{i}')
    result = model.score(x_test,y_test)
    print('model_score : ', result)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_pred,y_test)
    print('r2_score : ', r2)



'''
*no scaler


*MinMax scaler

*Standard scaler 

 '''