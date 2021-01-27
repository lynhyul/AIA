import numpy as np
from sklearn.datasets import load_diabetes
import tensorflow as tf

from sklearn.metrics import accuracy_score,r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')


dataset = load_diabetes()
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
Deep learning (Tensorflow) Conv1d
RMSE:  44.9913009554782
R2:  0.6541214055043305

ARDRegression 의 정답률 :  0.6246343099042484
AdaBoostRegressor 의 정답률 :  0.5437738212678458
BaggingRegressor 의 정답률 :  0.5294039169832221
BayesianRidge 의 정답률 :  0.6136328935846161
CCA 의 정답률 :  0.593468913579405
DecisionTreeRegressor 의 정답률 :  0.19352482024319273
DummyRegressor 의 정답률 :  -0.0008999028790297459
ElasticNet 의 정답률 :  0.14464513002867552
ElasticNetCV 의 정답률 :  0.5998259244939173
ExtraTreeRegressor 의 정답률 :  -0.012028311360072985
ExtraTreesRegressor 의 정답률 :  0.6041562512039765
GammaRegressor 의 정답률 :  0.08743878225684165
GaussianProcessRegressor 의 정답률 :  -17.121366938068032
GeneralizedLinearRegressor 의 정답률 :  0.0883736310408949
GradientBoostingRegressor 의 정답률 :  0.5748849763883478
HistGradientBoostingRegressor 의 정답률 :  0.6043077817229014
HuberRegressor 의 정답률 :  0.620997594864916
IsotonicRegression 은 없는놈!
KNeighborsRegressor 의 정답률 :  0.47102341020188654
KernelRidge 의 정답률 :  0.6166533203906133
Lars 의 정답률 :  0.6192841902078183
LarsCV 의 정답률 :  0.6203389104144862
Lasso 의 정답률 :  0.6121820117265979
LassoCV 의 정답률 :  0.6181268840587112
LassoLars 의 정답률 :  0.4811047109188421
LassoLarsCV 의 정답률 :  0.6203389104144862
LassoLarsIC 의 정답률 :  0.61670606940452
LinearRegression 의 정답률 :  0.619284190207818
LinearSVR 의 정답률 :  0.24352897567920362
MLPRegressor 의 정답률 :  -0.5854192370097544
MultiOutputRegressor 은 없는놈!
MultiTaskElasticNet 은 없는놈!
MultiTaskElasticNetCV 은 없는놈!
MultiTaskLasso 은 없는놈!
MultiTaskLassoCV 은 없는놈!
NuSVR 의 정답률 :  0.1575642533169357
OrthogonalMatchingPursuit 의 정답률 :  0.5233691572211906
OrthogonalMatchingPursuitCV 의 정답률 :  0.6177402574527012
PLSCanonical 의 정답률 :  -1.304467850784051
PLSRegression 의 정답률 :  0.6028083795809528
PassiveAggressiveRegressor 의 정답률 :  0.578151553596684
PoissonRegressor 의 정답률 :  0.5934819556685069
RANSACRegressor 의 정답률 :  -0.1475508986769969
RadiusNeighborsRegressor 의 정답률 :  0.1709310397415953
RandomForestRegressor 의 정답률 :  0.571136390820177
RegressorChain 은 없는놈!
Ridge 의 정답률 :  0.6113540839765157
RidgeCV 의 정답률 :  0.6113540839765175
SGDRegressor 의 정답률 :  0.6057644322579494
SVR 의 정답률 :  0.1503897394975161
StackingRegressor 은 없는놈!
TheilSenRegressor 의 정답률 :  0.628254621569313
TransformedTargetRegressor 의 정답률 :  0.619284190207818
TweedieRegressor 의 정답률 :  0.0883736310408949
VotingRegressor 은 없는놈!
_SigmoidCalibration 은 없는놈!
'''