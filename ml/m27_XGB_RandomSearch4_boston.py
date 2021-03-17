#다중분류
# Model = RandomForestClassifier

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score,r2_score

from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

import datetime

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')

#1. data
#x, y = load_iris(return_X_y=True)

dataset = load_boston()
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
    {"n_estimators":[100,200,300], "learning_rate":[0.01,0.03]
    ,"max_depth":[4,5,6]},
     {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3]
    ,"max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1] },
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3]
    ,"max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1],
     "colsample_bylevel":[0.6,0.7,0.9] }
]

#2. modeling

# model = SVC()
model = RandomizedSearchCV(XGBRegressor(n_jobs = 2), parameters, cv=kfold)

#3. fit
model.fit(x_train, y_train)

#4. evoluate, predict
print("best parameter : ", model.best_estimator_)



y_pred = model.predict(x_test) # grid serach
print("best score : ",r2_score(y_test, y_pred))
# 1.0

date_now2 = datetime.datetime.now()
date_time = date_now2.strftime("%m월%d일_%H시%M분%S초")
print("End time: ",date_time)
print("걸린시간 : ",(date_now2-date_now1))

'''
start time:  02월01일_11시45분35초
best parameter :  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.3, max_delta_step=0, max_depth=5,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=200, n_jobs=2, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
best score :  0.8667315909055684
End time:  02월01일_11시45분39초
걸린시간 :  0:00:03.868639
'''