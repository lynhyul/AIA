# 실습
# 1. 상단모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성
# 최적의 R2 값과 피처임포턴스 구할것

# 2. 위 쓰레드 값으로 SelectFromModel 을 구해서
# 최적의 피처 개수를 구할 것

# 3. 위 피처 개수로 데이터(피처)를 수정해서
# 그리드서치 또는 랜덤서치 적용하여
# 최적의 R2 구할 것

# 1번값과 2번값 비교

from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

x, y = load_boston(return_X_y= True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = True, random_state = 66
)

#일단 feature_importance를 뽑자 , xg부스터는 gpu사용
xgb = XGBRegressor(tree_method='gpu_hist', gpu_id=0)

parameters = {
    'max_depth' : [2, 4, 6, -1],
    'min_child_weight' : [1, 2, 4, -1],
    'eta' : [0.3, 0.1, 0.01, 0.5]
}

model = RandomizedSearchCV(xgb, param_distributions= parameters, cv = 5, )

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2 : ', score)

thresholds = np.sort(model.best_estimator_.feature_importances_) # 이렇게 하면 fi 값들이 정렬되어 나온다!

print(thresholds)
# [1.9896124e-04 2.3081598e-03 9.5453663e-03 1.1583471e-02 1.3474984e-02
#  1.7375207e-02 2.0021910e-02 2.6737049e-02 4.7078669e-02 6.6335171e-02
#  6.9025055e-02 2.6033255e-01 4.5598343e-01]

tmp = 0
tmp2 = [0,0]
for thresh in thresholds:
    
    selection = SelectFromModel(model.best_estimator_, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)
    # 최적의 Threshold만 찾아본다.
    selection_model = XGBRegressor(n_jobs = 8, tree_method='gpu_hist', gpu_id=0)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    # 새로운 점수가 기존의 점수보다 높을경우, tmp[], tmp2[,]의 값을 갱신한다.
    score = r2_score(y_test, y_predict)
    if score > tmp :
        tmp = score
        tmp2[0] = thresh
        tmp2[1] = select_x_train.shape[1]

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100))
    print(f'Best Score so far : {tmp*100}%')
    print('Best Threshold : ', tmp2[0])

print('=========================================================================================')
print(f'Best Threshold : {tmp2[0]}, n = {tmp2[1]}')

selection = SelectFromModel(model.best_estimator_, threshold = tmp2[0], prefit = True)

select_x_train = selection.transform(x_train)

selection_model = RandomizedSearchCV(xgb, parameters, cv =5)
selection_model.fit(select_x_train, y_train)

select_x_test = selection.transform(x_test)
y_predict = selection_model.predict(select_x_test)

score = r2_score(y_test, y_predict)

print('=========================================================================================')
print(f'최종 R2 score : {score*100}%, n = {tmp2[1]}일때!!')
print('=========================================================================================')
print(f'1번 점수 : {tmp*100}%\n2번 점수 : {score*100}%')
print('=========================================================================================')