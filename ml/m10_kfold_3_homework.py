# train, test 나눈 다음에 train만 발리데이션 하지 말고,
# kfold한 후에 train_Test_split 사용

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')
# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

kfold = KFold(n_splits=5, shuffle=True)
# 2. 모델 구성

model = [LinearSVC(), SVC(), KNeighborsClassifier(),RandomForestClassifier(),DecisionTreeClassifier()]
i = 1
for train_index, test_index in kfold.split(x):
    print(str(i)+'번째 kfold split')
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=32)
    for j in model:
        score = cross_val_score(j, x_train, y_train, cv=kfold)
        print('score :',score, f' ({j})')
    i += 1

# 1번째 kfold split
# score : [0.95       0.84210526 0.94736842 0.94736842 1.        ]  (LinearSVC())
# score : [0.95       0.94736842 0.89473684 0.89473684 1.        ]  (SVC())
# score : [1.         0.94736842 0.94736842 0.89473684 0.94736842]  (KNeighborsClassifier())
# score : [0.95       0.89473684 0.94736842 0.94736842 0.94736842]  (RandomForestClassifier())
# score : [0.95       1.         0.84210526 0.94736842 0.89473684]  (DecisionTreeClassifier())

# 2번째 kfold split
# score : [1.         1.         0.94736842 0.89473684 1.        ]  (LinearSVC())
# score : [1.         1.         0.94736842 0.84210526 1.        ]  (SVC())
# score : [1.         0.94736842 0.89473684 1.         1.        ]  (KNeighborsClassifier())
# score : [0.95       1.         0.94736842 1.         1.        ]  (RandomForestClassifier())
# score : [1.         0.94736842 0.94736842 0.78947368 1.        ]  (DecisionTreeClassifier())

# 3번째 kfold split
# score : [1.         0.89473684 1.         0.94736842 0.94736842]  (LinearSVC())
# score : [0.95       0.94736842 1.         1.         0.89473684]  (SVC())
# score : [0.95       0.94736842 1.         0.94736842 0.89473684]  (KNeighborsClassifier())
# score : [0.9        0.94736842 1.         0.89473684 1.        ]  (RandomForestClassifier())
# score : [0.9        0.94736842 1.         0.89473684 1.        ]  (DecisionTreeClassifier())

# 4번째 kfold split
# score : [0.8        0.94736842 1.         0.94736842 1.        ]  (LinearSVC())
# score : [0.95       1.         0.94736842 0.89473684 0.84210526]  (SVC())
# score : [0.95       0.94736842 0.89473684 1.         0.94736842]  (KNeighborsClassifier())
# score : [1.         0.84210526 1.         0.94736842 0.84210526]  (RandomForestClassifier())
# score : [0.95       0.84210526 0.94736842 0.94736842 1.        ]  (DecisionTreeClassifier())

# 5번째 kfold split
# score : [1.         0.78947368 0.94736842 0.89473684 1.        ]  (LinearSVC())
# score : [0.95       0.89473684 0.94736842 0.89473684 0.89473684]  (SVC())
# score : [0.95       0.94736842 0.94736842 1.         0.89473684]  (KNeighborsClassifier())
# score : [0.95       0.89473684 1.         0.89473684 0.94736842]  (RandomForestClassifier())
# score : [0.95       0.94736842 0.94736842 0.94736842 0.94736842]  (DecisionTreeClassifier())