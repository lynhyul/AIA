import numpy as np
from sklearn.datasets import load_boston
import tensorflow as tf

from sklearn.metrics import accuracy_score,r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')


dataset = load_boston()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, 
                                                    random_state=110)

allAlgorithms = all_estimators(type_filter='regressor')

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

for(name, algorithm) in allAlgorithms :
    try:    
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred))
    except :
        print(name, '은 없는놈!')
#shift Tap : reverse before push Tab key

import sklearn
print(sklearn.__version__)  # 0.23.2

'''
Deep learning (Tensorflow) CNN
RMSE :  2.804320026416158
R2 :  0.9048113443782645

ARDRegression 의 정답률 :  0.7169803398979304
AdaBoostRegressor 의 정답률 :  0.8230351612639473
BaggingRegressor 의 정답률 :  0.860465720153094
BayesianRidge 의 정답률 :  0.715903539862249
CCA 의 정답률 :  0.6074916000945052
DecisionTreeRegressor 의 정답률 :  0.7908713371527298
DummyRegressor 의 정답률 :  -0.0021710926187736845
ElasticNet 의 정답률 :  0.19541397285474682
ElasticNetCV 의 정답률 :  0.7184202281331593
ExtraTreeRegressor 의 정답률 :  0.6656069913312814
ExtraTreesRegressor 의 정답률 :  0.8776961792468849
GammaRegressor 의 정답률 :  0.21805913123165255
GaussianProcessRegressor 의 정답률 :  -0.7044113694875991
GeneralizedLinearRegressor 의 정답률 :  0.22068127740049437
GradientBoostingRegressor 의 정답률 :  0.8794049252443972
HistGradientBoostingRegressor 의 정답률 :  0.8624573907920858
HuberRegressor 의 정답률 :  0.7093108244674469
IsotonicRegression 은 없는놈!
KNeighborsRegressor 의 정답률 :  0.7380311256592333
KernelRidge 의 정답률 :  0.6106714912866072
Lars 의 정답률 :  0.7123045541390602
LarsCV 의 정답률 :  0.7140883111289231
Lasso 의 정답률 :  0.30399982735006104
LassoCV 의 정답률 :  0.7155506342895654
LassoLars 의 정답률 :  -0.0021710926187736845
LassoLarsCV 의 정답률 :  0.7165959987032173
LassoLarsIC 의 정답률 :  0.7154549349989464
LinearRegression 의 정답률 :  0.7139513256060439
LinearSVR 의 정답률 :  0.6117961091335635
MLPRegressor 의 정답률 :  0.33976785366053597
MultiOutputRegressor 은 없는놈!
MultiTaskElasticNet 은 없는놈!
MultiTaskElasticNetCV 은 없는놈!
MultiTaskLasso 은 없는놈!
MultiTaskLassoCV 은 없는놈!
NuSVR 의 정답률 :  0.6285656434814244
OrthogonalMatchingPursuit 의 정답률 :  0.5640314101285199
OrthogonalMatchingPursuitCV 의 정답률 :  0.6835353547486629
PLSCanonical 의 정답률 :  -3.2799371799704566
PLSRegression 의 정답률 :  0.700013771067252
PassiveAggressiveRegressor 의 정답률 :  0.6612143160874239
PoissonRegressor 의 정답률 :  0.6581185156330878
RANSACRegressor 의 정답률 :  0.5040830638108563
RadiusNeighborsRegressor 의 정답률 :  0.36853746141471155
RandomForestRegressor 의 정답률 :  0.854459318598916
RegressorChain 은 없는놈!
Ridge 의 정답률 :  0.7191508746921484
RidgeCV 의 정답률 :  0.7148645418060726
SGDRegressor 의 정답률 :  0.7008836843090378
SVR 의 정답률 :  0.6512917332102218
StackingRegressor 은 없는놈!
TheilSenRegressor 의 정답률 :  0.7286626702117922
TransformedTargetRegressor 의 정답률 :  0.7139513256060439
TweedieRegressor 의 정답률 :  0.22068127740049437
VotingRegressor 은 없는놈!
_SigmoidCalibration 은 없는놈!
'''
