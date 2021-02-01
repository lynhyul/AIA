# 실습
# pca를 통해 0.95 이상인거 몇개?
# pca 다 집어넣고 확인!!

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split as ts
from xgboost import XGBClassifier
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x = np.append(x_train,x_test,axis=0)

print(x_train.shape)  # 60000,28,28
print(x_test.shape)   # 10000, 28,28

x_train = x_train.reshape(x_train.shape[0],28*28)
x_test = x_test.reshape(x_test.shape[0],28*28)

pca = PCA()
pca.fit(x_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
# print(cumsum)   
'''
d : 154
'''

d = np.argmax(cumsum >= 1.0)+1
print("cumsum >= 0.95", cumsum>=1.0)
print("d :", d)

# pca = PCA(n_components=154)
# x_train = pca.fit_transform(x_train)  # merge fit,transform
# x_test = pca.transform(x_test)

# model = XGBClassifier(n_jobs=2,use_label_encoder=False)

# model.fit(x_train,y_train,eval_metric='logloss')

# acc = model.score(x_test,y_test)
# print("acc : ",acc)