import numpy as np
import pandas as pd

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline, make_pipeline

import warnings
warnings.filterwarnings('ignore')

import datetime

dataset = load_boston()
x = dataset.data    # 150,4
y = dataset.target  # 150,


x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=110,
shuffle = True, train_size = 0.8 )

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

parameters = [
    {"mod__n_estimators" : [100,200,300]},  
    {'mod__max_depth' : [-1,2,4,6]},    # 깊이
    {'mod__min_samples_leaf' : [3,5,7,10]},
    {'mod__min_samples_split' : [2,3,5,10]},
    {'mod__n_jobs' : [-1]}   # cpu를 몇 개 쓸것인지
]

pipe = Pipeline([('scaler', MinMaxScaler()),('mod', RandomForestRegressor())])
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier()) 


  
models = [GridSearchCV,RandomizedSearchCV]
scalers = [MinMaxScaler(),StandardScaler()]


for i in scalers:
    scaler = i       
    pipe = Pipeline([('scaler', i),('mod', RandomForestRegressor())])
    for j in models :
        model = j(pipe, parameters, cv=5)
        date_now1 = datetime.datetime.now()
        model.fit(x_train,y_train)
        date_now2 = datetime.datetime.now()
        results = model.score(x_test,y_test)
        print("\n",f'{j.__name__}')
        print(f'{i} : ',results)
        print("걸린시간 : ",(date_now2-date_now1))



'''
 GridSearchCV
MinMaxScaler() :  0.8572145994577856
걸린시간 :  0:00:15.815237

 RandomizedSearchCV
MinMaxScaler() :  0.8567020513656733
걸린시간 :  0:00:07.348193

 GridSearchCV
StandardScaler() :  0.8560341682849005
걸린시간 :  0:00:13.044228

 RandomizedSearchCV
StandardScaler() :  0.8505354791033841
걸린시간 :  0:00:09.234971
'''