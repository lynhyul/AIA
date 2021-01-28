#다중분류
# MinMax scaler

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')

#1. data
#x, y = load_iris(return_X_y=True)

dataset = load_wine()


x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

dataset = dataset.to_numpy()


print(x.shape)      # (150,4)
print(y.shape)      # (150,)

print(x[:5])
print(y)


print(y)
print(x.shape)  # (150,4)
print(y.shape)  # (150,3)

import datetime

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=110,
shuffle = True, train_size = 0.8 )

kfold = KFold(n_splits=5, shuffle=True)

date_now1 = datetime.datetime.now()
date_time = date_now1.strftime("%m월%d일_%H시%M분%S초")
print("start time: ",date_time)

parameters = [
    {"n_estimators" : [100,200,300]},
    {'max_depth' : [-1,2,4,6,8,10]},
    {'min_samples_leaf' : [3,5,7,10,12,14]},
    {'min_samples_split' : [2,3,5,10,12,14]},
    {'n_jobs' : [-1,2,4]}
]


#2. modeling

# model = SVC()
# model = GridSearchCV(SVC(), parameters, cv=kfold)
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold)
#3. fit
model.fit(x_train, y_train)

#4. evoluate, predict
print("best parameter : ", model.best_estimator_)
# best parameter :  SVC(C=1, kernel='linear')


y_pred = model.predict(x_test) # grid serach
print("best score : ",accuracy_score(y_test, y_pred))
# 1.0

date_now2 = datetime.datetime.now()
date_time = date_now2.strftime("%m월%d일_%H시%M분%S초")
print("End time: ",date_time)
print("걸린시간 : ",(date_now2-date_now1))


'''
Deep learning
loss :  [0.01415738184005022, 1.0]
best parameter :  SVC(C=100, gamma=0.001)
best score :  1.0
'''