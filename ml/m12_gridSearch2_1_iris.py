#다중분류
# Model = RandomForestClassifier

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import datetime

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')

#1. data
#x, y = load_iris(return_X_y=True)

dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape)      # (150,4)
print(y.shape)      # (150,)

print(x[:5])
print(y)


print(y)
print(x.shape)  # (150,4)
print(y.shape)  # (150,3)

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=110,
shuffle = True, train_size = 0.8 )

kfold = KFold(n_splits=5, shuffle=True)


date_now1 = datetime.datetime.now()
date_time = date_now1.strftime("%m월%d일_%H시%M분%S초")
print("start time: ",date_time)

parameters = [
    # {"n_estimators" : [300],
    # 'max_depth' : [-1],
    # 'min_samples_leaf' : [-1,2,14],
    # 'min_samples_split' : [-1,2,14],
    # 'n_jobs' :[-1]}
    {"n_estimators" : [100,200,300]},  
    {'max_depth' : [-1,2,4,6,8,10]},    # 깊이
    {'min_samples_leaf' : [3,5,7,10,12,14]},
    {'min_samples_split' : [2,3,5,10,12,14]},
    {'n_jobs' : [-1,2,4]}   # cpu를 몇 개 쓸것인지
]

#2. modeling

# model = SVC()
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold)

#3. fit
model.fit(x_train, y_train)

#4. evoluate, predict
print("best parameter : ", model.best_estimator_)



y_pred = model.predict(x_test) # grid serach
print("best score : ",accuracy_score(y_test, y_pred))
# 1.0

date_now2 = datetime.datetime.now()
date_time = date_now2.strftime("%m월%d일_%H시%M분%S초")
print("End time: ",date_time)
print("걸린시간 : ",(date_now2-date_now1))



'''
start time:  01월28일_16시52분53초
best parameter :  RandomForestClassifier(max_depth=4)
best score :  0.9666666666666667
End time:  01월28일_16시53분11초
걸린시간 :  0:00:18.401929

 '''