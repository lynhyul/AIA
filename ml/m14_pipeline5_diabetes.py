import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC,SVC

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline, make_pipeline

import warnings
warnings.filterwarnings('ignore')

dataset = load_diabetes()
x = dataset.data    # 150,4
y = dataset.target  # 150,


x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=110,
shuffle = True, train_size = 0.8 )

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
# model = Pipeline([('scaler', MinMaxScaler()),('model', SVC())])  
# 전처리까지 합치는 과정, ' '안에 들어가는 이름은 아무거나 해줘도 상관은없다
models = [KNeighborsRegressor(),DecisionTreeRegressor(),RandomForestRegressor()]
scalers = [MinMaxScaler(),StandardScaler()]
for i in models:
    model = i
    print(f'\n{i}')
    for j in scalers:
        scaler = j
        print(f'{j}')
        model = make_pipeline(j, i)
        model.fit(x_train,y_train)
        results = model.score(x_test,y_test)
        print(results)

'''
KNeighborsRegressor()
MinMaxScaler()
0.47102341020188654
StandardScaler()
0.47058529107208824

DecisionTreeRegressor()
MinMaxScaler()
0.13083461890530046
StandardScaler()
0.17142879454271187

RandomForestRegressor()
MinMaxScaler()
0.5968954787278447
StandardScaler()
0.5720394436216236
'''