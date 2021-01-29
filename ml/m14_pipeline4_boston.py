import numpy as np
import pandas as pd

from sklearn.datasets import load_boston

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

dataset = load_boston()
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
models = [KNeighborsRegressor(),RandomForestRegressor(),DecisionTreeRegressor()]
scalers = [MinMaxScaler(),StandardScaler()]
for i in models:
    model = i
    print(f'\n{i}')
    for j in scalers:
        scaler = j       
        model = make_pipeline(j, i)
        model.fit(x_train,y_train)
        results = model.score(x_test,y_test)
        print(f'{j} : ',results)

'''
KNeighborsRegressor()
MinMaxScaler() :  0.7380311256592333
StandardScaler() :  0.7545891164317223

RandomForestRegressor()
MinMaxScaler() :  0.8549459072741515
StandardScaler() :  0.8518589414491722

DecisionTreeRegressor()
MinMaxScaler() :  0.8032534568083837
StandardScaler() :  0.7066685452118009
'''