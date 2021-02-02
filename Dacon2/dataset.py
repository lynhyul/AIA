import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score,RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

train = pd.read_csv('../data/csv/practice/train.csv')
test = pd.read_csv('../data/csv/practice/test.csv')



temp = pd.DataFrame(train)
test_df = pd.DataFrame(test)

x = temp.iloc[:,3:]/255
y = temp.iloc[:,[1]]
x_test = test_df.iloc[:,2:]/255

x = x.to_numpy()
y = y.to_numpy()
x_pred = x_test.to_numpy()

print(x.shape)
print(y.shape)



# pca = PCA(n_components=420)
# x = pca.fit_transform(x)
# x_pred = pca.transform(x_pred)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,
                                            random_state =110) 


# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)


# print(cumsum)   
# '''
# d : 277
# '''

# d = np.argmax(cumsum >= 0.999)+1
# print("cumsum >= 0.95", cumsum>=0.999)
# print("d :", d)




# 2. 모델

# model = XGBClassifier(n_estimators=1000, n_jobs=8, learning_rate =0.017 ,use_label_encoder=False)


'''
score = cross_val_score(model ,x_train, y_train, cv=kfold)
print(score)
'''


kfold = KFold(n_splits=5, shuffle=True)


# parameters = [
#     {"n_estimators":[100,200,300], "learning_rate":[0.01,0.03]
#     ,"max_depth":[4,5,6]},
#      {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3]
#     ,"max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1] },
#     {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3]
#     ,"max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1],
#      "colsample_bylevel":[0.6,0.7,0.9] }
# ]

#2. modeling

# model = SVC()
model = RandomizedSearchCV(model = XGBClassifier(n_estimators=1000, n_jobs=8, learning_rate =0.017 ,use_label_encoder=False)
                            , cv=kfold)
#3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric='mlogloss',early_stopping_rounds=20,
eval_set=[(x_train,y_train),(x_test,y_test)])

#4. 평가 예측
acc = model.score(x_test,y_test)

#print(model.feature_importances_)
print('acc : ', acc)

y_pred = model.predict(x_pred)


# print(y_train.shape)
# print(x_train.shape)
# print(x_test.shape)
