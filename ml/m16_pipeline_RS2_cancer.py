import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline, make_pipeline

import warnings
warnings.filterwarnings('ignore')

import datetime

dataset = load_breast_cancer()
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

models = [GridSearchCV,RandomizedSearchCV]
scalers = [MinMaxScaler(),StandardScaler()]


for i in scalers:
    scaler = i       
    pipe = Pipeline([('scaler', i),('mod', RandomForestClassifier())])
    for j in models :
        model = j(pipe, parameters, cv=5)
        date_now1 = datetime.datetime.now()
        model.fit(x_train,y_train)
        date_now2 = datetime.datetime.now()
        results = model.score(x_test,y_test)
        print(f'{j.__name__}')
        print(f'{i} : ',results)
        print("걸린시간 : ",(date_now2-date_now1))



'''
GridSearchCV
MinMaxScaler() :  0.9649122807017544
걸린시간 :  0:00:12.668875
RandomizedSearchCV
MinMaxScaler() :  0.9824561403508771
걸린시간 :  0:00:07.439054
GridSearchCV
StandardScaler() :  0.9736842105263158
걸린시간 :  0:00:10.394949
RandomizedSearchCV
StandardScaler() :  0.956140350877193
걸린시간 :  0:00:06.488935
'''