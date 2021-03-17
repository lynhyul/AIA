import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline, make_pipeline

import warnings
warnings.filterwarnings('ignore')

dataset = load_iris()
x = dataset.data    # 150,4
y = dataset.target  # 150,


x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=110,
shuffle = True, train_size = 0.8 )

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

parameters = [
    {"svc__C" : [1,10,100,1000], "svc__kernel":["linear"]},   # kernel => activation function
    {"svc__C" : [1,10,100], "svc__kernel":["rbf"] , "svc__gamma": [0.001,0.0001] }, # gamma => lr function
    {"svc__C" : [1,10,100,1000],"svc__kernel":["sigmoid"], "svc__gamma":[0.001,0.0001] }
]

# pipe = Pipeline([('scaler', MinMaxScaler()),('mod', SVC())])
pipe = make_pipeline(MinMaxScaler(), SVC()) 
# Pipeline(steps=[('minmaxscaler', MinMaxScaler()), ('svc', SVC())])

  
model = GridSearchCV(pipe, parameters, cv=5)

model.fit(x_train,y_train)

results = model.score(x_test,y_test)
print(results)

# models = [GridSearchCV(pipe, parameters, cv=5), RandomizedSearchCV(pipe, parameters, cv=5)]
# names = [GridSearchCV, RandomizedSearchCV]
# scalers = [MinMaxScaler(),StandardScaler()]

# for i in models:
#     model = i
#     for j in scalers:
#         scaler = j       
#         pipe = make_pipeline(j, SVC()) 
#         model.fit(x_train,y_train)
#         results = model.score(x_test,y_test)
#         print(f'{j} : ',results)

'''
0.9666666666666667
'''