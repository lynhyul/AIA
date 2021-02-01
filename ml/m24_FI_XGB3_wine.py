from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import numpy as np
import warnings
import datetime


from xgboost import XGBClassifier

# 1. 데이터
dataset = load_wine()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=32
)

date_now1 = datetime.datetime.now()
# model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(n_jobs=4)


model.fit(x_train,y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
date_now2 = datetime.datetime.now()
print("걸린시간 : ",(date_now2-date_now1))


fi = model.feature_importances_
fi = pd.DataFrame(fi).quantile(q=0.0)
fi = fi.to_numpy()
print(fi)
print("개선 이전의 acc :", acc)

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
cl = df.columns
new_data=[]
for i in range(len(cl)):
    if model.feature_importances_[i] > fi:
       new_data.append(df.iloc[:,i])
new_data = pd.concat(new_data, axis=1)
# print(new_data.columns)

new_data1 = new_data.to_numpy()
# print(new_data1.shape)

x2_train, x2_test, y2_train, y2_test = train_test_split(new_data1, dataset.target, train_size=0.8, random_state=32)
# model2 = GradientBoostingClassifier(max_depth=4)
model2 = XGBClassifier(n_jobs=-1)   
model2.fit(x2_train,y2_train)
acc2 = model2.score(x2_test, y2_test)
print("개선 이후의 acc :", acc2)


import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model) :
    n_features = new_data1.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
            align = 'center')
    plt.yticks(np.arange(n_features), new_data.columns)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

plot_feature_importances_dataset(model2)
plt.show()

'''
개선 이전의 acc : 0.9722222222222222
[10:39:56] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.3.0/src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
개선 이후의 acc : 0.9722222222222222
'''