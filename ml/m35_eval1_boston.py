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
model = XGBRegressor(n_estimators=10, learning_rate=0.017,n_jobs=8)

#3. fit
model.fit(x_train,y_train, verbose=1, eval_metric='rmse',
            eval_set=[(x_train,y_train),(x_test,y_test)])
aaa = model.score(x_test,y_test)
print("aaa : ", aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ",r2)

result = model.evals_result()
print(result)

'''
aaa :  -5.947174881414525
r2 :  -5.947174881414525
{'validation_0': OrderedDict([('rmse', [23.557255, 23.177334, 22.803787, 22.43652, 22.075443, 
21.72043, 21.371067, 21.028139, 20.690361, 20.358854])]), 
'validation_1': OrderedDict([('rmse', [23.211592, 22.842552, 22.478613, 22.124048, 21.773285, 
21.429035, 21.092672, 20.760904, 20.433691, 20.111238])])}
'''