import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# 1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)


from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델 구성
models =[LinearRegression(), RandomForestRegressor(), DecisionTreeRegressor(), KNeighborsRegressor()]
for i in models:
    model = i
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    result = model.score(x_test, y_test)
    print(f'\n{i}')
    print("model.score :",result)
    acc = r2_score(y_test, y_pred)
    print("r2_score :",acc)

'''
Deep learning (Tensorflow)
R2 :  0.8115398445785679

*no scaler

LinearRegression()
model.score : 0.6314503088342935
r2_score : 0.6314503088342935

RandomForestRegressor()
model.score : 0.808093546962581
r2_score : 0.808093546962581

DecisionTreeRegressor()
model.score : 0.5878838446609981
r2_score : 0.5878838446609981

KNeighborsRegressor()
model.score : 0.4167927961042365
r2_score : 0.4167927961042365


*MinMax scaler

LinearRegression()
model.score : 0.7676425838752229
r2_score : 0.7676425838752229
RandomForestRegressor()
model.score : 0.8467992256569992
r2_score : 0.8467992256569992

DecisionTreeRegressor()
model.score : 0.773650084303442
r2_score : 0.773650084303442

KNeighborsRegressor()
model.score : 0.6785925428737792
r2_score : 0.6785925428737792


LinearRegression()
model.score : 0.7625350758673459
r2_score : 0.7625350758673459

RandomForestRegressor()
model.score : 0.8705416791784929
r2_score : 0.8705416791784929

DecisionTreeRegressor()
model.score : 0.6914162848253556
r2_score : 0.6914162848253556

KNeighborsRegressor()
model.score : 0.7972280204681221
r2_score : 0.7972280204681221


*Standard scaler

LinearRegression()
model.score : 0.7676425838752229
r2_score : 0.7676425838752229

RandomForestRegressor()
model.score : 0.8467992256569992
r2_score : 0.8467992256569992

DecisionTreeRegressor()
model.score : 0.773650084303442
r2_score : 0.773650084303442

KNeighborsRegressor()
model.score : 0.6785925428737792
r2_score : 0.6785925428737792

 '''