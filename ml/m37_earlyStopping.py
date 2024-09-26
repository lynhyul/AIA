from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

x,y = load_boston(return_X_y=True)
dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size = 0.8, shuffle = True, random_state = 110)


#2. modeling
model = XGBRegressor(n_estimators=10000, learning_rate=0.017,n_jobs=8)

#3. fit
model.fit(x_train,y_train, verbose=1, eval_metric=['rmse','logloss','mae'],early_stopping_rounds=10,
            eval_set=[(x_train,y_train),(x_test,y_test)])   #2개이상의 metrics는 마지막metrics로 적용
aaa = model.score(x_test,y_test)
print("aaa : ", aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ",r2)

# result = model.evals_result()
# print(result)

'''
....
[346]   validation_0-rmse:0.79110       validation_0-logloss:-790.65759 validation_0-mae:0.58544        validation_1-rmse:2.86272       validation_1-logloss:-803.75574 validation_1-mae:2.08831
[347]   validation_0-rmse:0.78604       validation_0-logloss:-790.65759 validation_0-mae:0.58198        validation_1-rmse:2.86144       validation_1-logloss:-803.75574 validation_1-mae:2.08740
[348]   validation_0-rmse:0.78297       validation_0-logloss:-790.65759 validation_0-mae:0.57970        validation_1-rmse:2.86166       validation_1-logloss:-803.75574 validation_1-mae:2.08827
[349]   validation_0-rmse:0.78090       validation_0-logloss:-790.65759 validation_0-mae:0.57798        validation_1-rmse:2.86105       validation_1-logloss:-803.75574 validation_1-mae:2.08779
[350]   validation_0-rmse:0.77823       validation_0-logloss:-790.65759 validation_0-mae:0.57578        validation_1-rmse:2.86006       validation_1-logloss:-803.75574 validation_1-mae:2.08735
aaa :  0.8590811459497115
r2 :  0.8590811459497115
'''