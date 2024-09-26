import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, shuffle = True, 
                                                    random_state=110)

kfold = KFold(n_splits=5, shuffle=True) #shuffle은 행을 섞는것.


allAlgorithms = all_estimators(type_filter='regressor')

for(name, algorithm) in allAlgorithms :
    try:    
        model = algorithm()

        scores = cross_val_score(model, x_train,y_train, cv = kfold)
        print(name, '의 정답률 : \n', scores)
    except :
        print(name, '은 없는놈!')
        

import sklearn
print(sklearn.__version__)  # 0.23.2


'''
Deep learning (Tensorflow) DNN
loss :  [0.01415738184005022, 1.0]

ARDRegression 의 정답률 : 
 [0.67538952 0.79354259 0.72881254 0.6135698  0.76428981]
AdaBoostRegressor 의 정답률 : 
 [0.65369503 0.87473978 0.84592755 0.85678656 0.88380994]
BaggingRegressor 의 정답률 : 
 [0.89780354 0.72615484 0.92849202 0.79989146 0.86256135]
BayesianRidge 의 정답률 :
 [0.74346835 0.74407291 0.48113971 0.78159837 0.71668273]
CCA 의 정답률 : 
 [0.37968834 0.70882543 0.65805576 0.74901519 0.65644999]
DecisionTreeRegressor 의 정답률 :
 [0.6451981  0.76192791 0.83478715 0.83719116 0.44851812]
DummyRegressor 의 정답률 : 
 [-0.00960706 -0.0089788  -0.01474331 -0.00493606 -0.0155991 ]
ElasticNet 의 정답률 :
 [0.67467228 0.7282931  0.60526945 0.63339838 0.7315547 ]
ElasticNetCV 의 정답률 : 
 [0.69451628 0.68835186 0.66253693 0.58991782 0.62648846]
ExtraTreeRegressor 의 정답률 :
 [0.71441052 0.80358099 0.57602448 0.69309555 0.74773786]
ExtraTreesRegressor 의 정답률 : 
 [0.89628643 0.87782308 0.85192474 0.82685873 0.92895975]
GammaRegressor 의 정답률 :
 [-0.00594413 -0.00025954 -0.01314518 -0.01522986 -0.00951799]
GaussianProcessRegressor 의 정답률 : 
 [-4.72437734 -6.66049722 -5.16054404 -4.85039502 -6.98171299]
GeneralizedLinearRegressor 의 정답률 : 
 [0.5572627  0.67052516 0.68717193 0.69362986 0.64543108]
GradientBoostingRegressor 의 정답률 : 
 [0.91256374 0.84769924 0.87119078 0.87161699 0.88612626]
HistGradientBoostingRegressor 의 정답률 : 
 [0.85077063 0.80130049 0.87762691 0.87825808 0.91932495]
HuberRegressor 의 정답률 : 
 [0.72985006 0.80437255 0.27515122 0.66980846 0.60646663]
IsotonicRegression 의 정답률 :
 [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 : 
 [0.6383344  0.64639159 0.51208606 0.48599092 0.47598369]
KernelRidge 의 정답률 : 
 [0.67177669 0.63889839 0.71752762 0.79924632 0.56807493]
Lars 의 정답률 :
 [0.60787427 0.72441971 0.684205   0.75789188 0.78428518]
LarsCV 의 정답률 : 
 [0.701771   0.71012488 0.78051887 0.70928126 0.64062103]
Lasso 의 정답률 :
 [0.71022544 0.66638593 0.71742628 0.66333369 0.55256807]
LassoCV 의 정답률 : 
 [0.68522395 0.65581651 0.7618314  0.58360443 0.70704683]
LassoLars 의 정답률 :
 [-0.01665862 -0.02069437 -0.00039461 -0.00272113 -0.01374312]
LassoLarsCV 의 정답률 : 
 [0.68109024 0.56888145 0.74535517 0.74577049 0.78556142]
LassoLarsIC 의 정답률 : 
 [0.62190612 0.54487149 0.77540092 0.75571643 0.77327786]
LinearRegression 의 정답률 :
 [0.59802707 0.8026428  0.60362853 0.5845829  0.81923082]
LinearSVR 의 정답률 : 
 [0.68355681 0.58236221 0.32475198 0.14750792 0.19096584]
MLPRegressor 의 정답률 : 
 [0.59455915 0.38511966 0.58012354 0.47631191 0.50034043]
MultiOutputRegressor 은 없는놈!
MultiTaskElasticNet 의 정답률 :
 [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 :
 [nan nan nan nan nan]
MultiTaskLasso 의 정답률 :
 [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 :
 [nan nan nan nan nan]
NuSVR 의 정답률 : 
 [0.07614283 0.30099144 0.22135792 0.279697   0.18004545]
OrthogonalMatchingPursuit 의 정답률 :
 [0.51446659 0.47194364 0.52455386 0.50174041 0.59921946]
OrthogonalMatchingPursuitCV 의 정답률 : 
 [0.76584929 0.72936245 0.67147159 0.62725508 0.63439933]
PLSCanonical 의 정답률 : 
 [-3.45768559 -4.09586893 -2.13267879 -2.12234327 -1.15570335]
PLSRegression 의 정답률 :
 [0.77533571 0.73206349 0.66863628 0.53910496 0.6938205 ]
PassiveAggressiveRegressor 의 정답률 :
 [-0.54133762  0.05258398  0.20745437  0.04047782 -1.54961954]
PoissonRegressor 의 정답률 : 
 [0.77444236 0.7845202  0.84212304 0.71781263 0.6600082 ]
RANSACRegressor 의 정답률 : 
 [0.58148207 0.37082017 0.05845547 0.60068309 0.27532239]
RadiusNeighborsRegressor 은 없는놈!
RandomForestRegressor 의 정답률 : 
 [0.84600911 0.91554589 0.80733845 0.8892989  0.88078202]
RegressorChain 은 없는놈!
Ridge 의 정답률 :
 [0.62318805 0.85160714 0.75571806 0.60924608 0.71696824]
RidgeCV 의 정답률 :
 [0.74060836 0.73565132 0.78639206 0.65717361 0.65527717]
SGDRegressor 의 정답률 : 
 [-2.99286265e+26 -2.26695043e+26 -7.30263763e+26 -4.72158623e+25
 -3.21608035e+26]
SVR 의 정답률 : 
 [0.08777763 0.18071787 0.21855864 0.16687252 0.24754214]
StackingRegressor 은 없는놈!
TheilSenRegressor 의 정답률 : 
 [0.64686444 0.5430548  0.7705079  0.76240438 0.69698665]
TransformedTargetRegressor 의 정답률 :
 [0.70265432 0.76315379 0.77786017 0.56099955 0.71156387]
TweedieRegressor 의 정답률 : 
 [0.75061043 0.60989875 0.47377926 0.68352902 0.62037105]
VotingRegressor 은 없는놈!
'''