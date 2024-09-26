from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

x,y = load_boston(return_X_y=True)
dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size = 0.8, shuffle = True, random_state = 66)


#2. modeling
model = XGBRegressor(n_estimators=100000, learning_rate=0.017,n_jobs=8,
                     tree_method = 'gpu_hist',
                     predictor='gpu_predictor',
                     gpu_id=1)  # gpu갯수 설정



#3. fit
model.fit(x_train,y_train, verbose=1, eval_metric=['rmse','logloss','mae'],early_stopping_rounds=10,
            eval_set=[(x_train,y_train),(x_test,y_test)])   #2개이상의 metrics는 마지막metrics로 적용
aaa = model.score(x_test,y_test)
print("model.score : ", aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ",r2)

result = model.evals_result()


