# 데이터는 diabetes, 모델은 RandomForest쓰고
# 파이프라인 엮어서 25번 돌리기


import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes

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

dataset = load_diabetes()
x = dataset.data    # 150,4
y = dataset.target  # 150,


# x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=110,
# shuffle = True, train_size = 0.8 )

# cv과정을 두 번 쓰려 할 때, test에 대한 데이터가 버려질 위험이 있으므로 따로 분리하지 않는다.

parameters = [
    {"mod__n_estimators" : [100,200,300]},  
    {'mod__max_depth' : [-1,2,4,6]},    # 깊이
    {'mod__min_samples_leaf' : [3,5,7,10]},
    {'mod__min_samples_split' : [2,3,5,10]},
    {'mod__n_jobs' : [-1]}   # cpu를 몇 개 쓸것인지
]

models = [GridSearchCV,RandomizedSearchCV]
scalers = [MinMaxScaler(),StandardScaler()]

kfold = KFold(n_splits=5, shuffle=True)



for i in scalers:
    scaler = i       
    pipe = Pipeline([('scaler', i),('mod', RandomForestRegressor())])
    for j in models :
        model = j(pipe, parameters, cv=kfold)
        date_now1 = datetime.datetime.now()
        score = cross_val_score(model, x, y, cv=kfold)
        date_now2 = datetime.datetime.now()
        print("\n",f'{j.__name__}')
        print(f'{i} : ', score)
        print("걸린시간 : ",(date_now2-date_now1))

'''
 GridSearchCV
MinMaxScaler() :  [0.50986301 0.43918207 0.49656031 0.42555571 0.47097104]
걸린시간 :  0:01:00.386790

 RandomizedSearchCV
MinMaxScaler() :  [0.32716404 0.53238929 0.21553789 0.52866004 0.4927737 ]
걸린시간 :  0:00:37.654398

 GridSearchCV
StandardScaler() :  [0.5261141  0.33789237 0.39616614 0.48724229 0.44707621]
걸린시간 :  0:00:57.819956

 RandomizedSearchCV
StandardScaler() :  [0.45335153 0.36150175 0.43316331 0.49268646 0.51051288]
걸린시간 :  0:00:38.191674
'''