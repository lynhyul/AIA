# 실습
# 1. 상단모델에 그리드서치 또는 랜덤서치로 튜닝한 모델을 구성
# 최적의 R2값과 피처임포턴스 구할것

# 2. 위 쓰레드 값으로 SelectFromModel을 구해서
# 최적의 피처 갯수를 구할 것

# 3. 위 피처 갯수로 데이터(피처)를 수정(삭제) 해서
# 그리드서치 또는 랜덤서치 적용하여
# 최적의 R2 구할 것

# 1번 값과 2번값 비교


from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,  KFold, cross_val_score,GridSearchCV,RandomizedSearchCV
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score
import datetime

x, y = load_boston(return_X_y=True) # 데이터 바로 가져오기

x_train, x_test , y_train, y_test = train_test_split(x,y, train_size = 0.8, shuffle = True, random_state= 66)

model = XGBRegressor()


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
model = XGBRegressor(n_jobs=8,use_label_encoder=False, tree_method='gpu_hist', gpu_id =0)

#3. fit
model.fit(x_train, y_train,eval_metric='mlogloss')

y_pred = model.predict(x_test) # grid serach
print("best score : ",r2_score(y_test, y_pred))
# 1.0




thresholds = np.sort(model.feature_importances_)
# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
#  0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
#  0.42848358] => 낮은 차순으로 정렬된다.

print(thresholds)

for thresh in thresholds :
    selection = SelectFromModel(model, threshold=thresh, prefit = True)

    select_x_train = selection.transform(x_train)   # x_train을 select 모드에 맞게 바꿔주겠다.
    print(select_x_train.shape)

    selection_model = GridSearchCV(XGBRegressor(n_jobs=8,use_label_encoder=False, tree_method='gpu_hist', gpu_id=0) , parameters, cv=kfold)
    selection_model.fit(select_x_train,y_train)
    
    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2 : %.2f%%" %(thresh, select_x_train.shape[1], score*100))
    print("best parameter : ", selection_model.best_estimator_)

    date_now2 = datetime.datetime.now()
    date_time = date_now2.strftime("%m월%d일_%H시%M분%S초")
    print("End time: ",date_time)
    print("걸린시간 : ",(date_now2-date_now1))