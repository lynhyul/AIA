from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

x,y = load_breast_cancer(return_X_y=True)
dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
    train_size = 0.8, shuffle = True, random_state = 110)


#2. modeling
model = XGBClassifier(n_estimators=10, learning_rate=0.017,n_jobs=8,use_label_encoder=False)

#3. fit
model.fit(x_train,y_train, verbose=1, eval_metric=['logloss','auc','aucpr'],    #2진분류 logloss
            eval_set=[(x_train,y_train),(x_test,y_test)],)
aaa = model.score(x_test,y_test)
print("aaa : ", aaa)

# y_pred = model.predict(x_test)
# acc = accuracy_score(x_test,y_test)
# print("acc : ",acc)

result = model.evals_result()
print(result)

'''
aaa :  0.9385964912280702
{'validation_0': OrderedDict([('logloss', [0.677995, 0.663533, 0.649533, 
0.635971, 0.622825, 0.610079, 0.597715, 0.585711, 0.574061, 0.562742])]), 
'validation_1': OrderedDict([('logloss', [0.678674, 0.664758, 0.651436, 0.638921, 0.626556,
 0.614325, 0.602684, 0.59153, 0.580267, 0.56978])])}
'''