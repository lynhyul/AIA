import numpy as np
import pandas as pd

train = pd.read_csv('C:/data/kaggle/input/tabular-playground-series-apr-2021/train.csv')
test = pd.read_csv('C:/data/kaggle/input/tabular-playground-series-apr-2021/test.csv')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

sample_submission = pd.read_csv('C:/data/kaggle/input/tabular-playground-series-apr-2021/sample_submission.csv')
y_train = train['Survived']
X_train = train.drop(['Survived','PassengerId'], axis=1)
X_test = test.drop(['PassengerId'], axis=1)
X_train['Cabin'] = X_train['Cabin'].fillna('X').map(lambda x: x[0:5].strip())
X_test['Cabin'] = X_test['Cabin'].fillna('X').map(lambda x: x[0:5].strip())

X_train['Name'] = X_train['Name'].fillna('X').map(lambda x: x.split(',')[0])
X_test['Name'] = X_test['Name'].fillna('X').map(lambda x: x.split(',')[0])

X_train[['Age','Fare']] = X_train[['Age','Fare']].fillna(X_train[['Age','Fare']].median())
X_test[['Age','Fare']] = X_test[['Age','Fare']].fillna(X_test[['Age','Fare']].median())

from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="most_frequent")

X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imp.fit_transform(X_test),  columns=X_test.columns)
from sklearn import preprocessing

enc = preprocessing.OrdinalEncoder()

enc.fit(X_train[['Sex','Cabin','Name','Ticket', 'Embarked']])
X_train[['Sex','Cabin','Name','Ticket','Embarked']] = enc.transform(X_train[['Sex','Cabin','Name','Ticket','Embarked']])


enc.fit(X_test[['Sex','Cabin','Name','Ticket','Embarked']])
X_test[['Sex','Cabin','Name','Ticket','Embarked']] = enc.transform(X_test[['Sex','Cabin','Name','Ticket','Embarked']])
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

scaler = preprocessing.StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

clf = HistGradientBoostingClassifier()
clf = AdaBoostClassifier(n_estimators=100)



clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_train, y_train, cv=5)
scores.mean()

predictions = clf.predict(X_test)
sample_submission['Survived'] = predictions
sample_submission.to_csv('submission1.csv',index=False)

