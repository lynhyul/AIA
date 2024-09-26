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
print("model.score : ", aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ",r2)

result = model.evals_result()
# print(result)

#(1) Save
import pickle
# pickle.dump(model, open('../data/xgb_save/m39.pikle.data','wb'))
# print('save complete')
import joblib
# joblib.dump(model,'../data/xgb_save/m40.joblib.data')


print("=========================pickle.Load======================")
#(2) Load
# model2 = pickle.load(open('../data/xgb_save/m39.pikle.data','rb'))

model2 = joblib.load('../data/xgb_save/m40.joblib.data')
r2_2 = model2.score(x_test,y_test)
print('r2_2 : ',r2_2)
