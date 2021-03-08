from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

x, y = load_boston(return_X_y=True) # 데이터 바로 가져오기

x_train, x_test , y_train, y_test = train_test_split(x,y, train_size = 0.8, shuffle = True, random_state= 66)

model = XGBRegressor()


model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("R2 : ",score)
# R2 :  0.9221188601856797

thresholds = np.sort(model.feature_importances_)
# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
#  0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
#  0.42848358] => 낮은 차순으로 정렬된다.
print(thresholds)

for thresh in thresholds :
    selection = SelectFromModel(model, threshold=thresh, prefit = True)

    select_x_train = selection.transform(x_train)   # x_train을 select 모드에 맞게 바꿔주겠다.
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_train,y_train)
    
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2 : %.2f%%" %(thresh, select_x_train.shape[1], score*100))
'''
(404, 13)
Thresh=0.001, n=13, R2 : 92.21%
(404, 12)
Thresh=0.004, n=12, R2 : 92.16%
(404, 11)
Thresh=0.012, n=11, R2 : 92.03%
(404, 10)
Thresh=0.012, n=10, R2 : 92.19%
(404, 9)
Thresh=0.014, n=9, R2 : 93.08%
(404, 8)
Thresh=0.015, n=8, R2 : 92.37%
(404, 7)
Thresh=0.018, n=7, R2 : 91.48%
(404, 6)
Thresh=0.030, n=6, R2 : 92.71%
(404, 5)
Thresh=0.042, n=5, R2 : 91.74%
(404, 4)
Thresh=0.052, n=4, R2 : 92.11%
(404, 3)
Thresh=0.069, n=3, R2 : 92.52%
(404, 2)
Thresh=0.301, n=2, R2 : 69.41%
(404, 1)
Thresh=0.428, n=1, R2 : 44.98%

feature importance가 가능한 머신러닝 계열에서 활용 가능.
'''
