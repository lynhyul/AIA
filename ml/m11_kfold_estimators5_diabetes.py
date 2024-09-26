import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_diabetes()
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
 [0.43640965 0.34556675 0.46674926 0.51968736 0.46749966]
AdaBoostRegressor 의 정답률 : 
 [0.38867473 0.45155751 0.29517652 0.30134661 0.4494967 ]
BaggingRegressor 의 정답률 : 
 [0.23058537 0.30589397 0.31917493 0.46840489 0.15140438]
BayesianRidge 의 정답률 :
 [0.4689468  0.41634257 0.49688867 0.38913337 0.47191456]
CCA 의 정답률 :
 [0.29534156 0.35411421 0.37540071 0.36206045 0.49627159]
DecisionTreeRegressor 의 정답률 : 
 [ 0.02309867 -0.38244565 -0.1475962   0.09645114 -0.31621969]
DummyRegressor 의 정답률 :
 [-0.0195602  -0.00436535 -0.09582553 -0.07807682 -0.02260481]
ElasticNet 의 정답률 : 
 [-0.03640558 -0.00277866  0.00756217 -0.01013284 -0.011229  ]
ElasticNetCV 의 정답률 : 
 [0.49146206 0.35984868 0.52389905 0.27552624 0.42630626]
ExtraTreeRegressor 의 정답률 :
 [-0.45319764  0.04925439  0.07025216 -0.55885136 -0.43773647]
ExtraTreesRegressor 의 정답률 : 
 [0.41195663 0.30757999 0.28957931 0.39146873 0.43046502]
GammaRegressor 의 정답률 :
 [ 0.002179   -0.01484066 -0.07723227 -0.06797328 -0.01132192]
GaussianProcessRegressor 의 정답률 : 
 [-14.77940713 -14.45535293  -4.95170774 -11.34255143 -15.25748693]
GeneralizedLinearRegressor 의 정답률 : 
 [ 0.00378123 -0.02910673  0.00240612 -0.00233848  0.00148329]
GradientBoostingRegressor 의 정답률 : 
 [0.23882659 0.33269791 0.43052472 0.30129513 0.40058078]
HistGradientBoostingRegressor 의 정답률 : 
 [0.18935261 0.28636537 0.33338846 0.44416122 0.22700784]
HuberRegressor 의 정답률 : 
 [0.39708006 0.35865891 0.61246142 0.405606   0.38032766]
IsotonicRegression 의 정답률 :
 [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 :
 [0.3168776  0.18554677 0.14427643 0.41095443 0.16661767]
KernelRidge 의 정답률 : 
 [-3.13151774 -3.37802136 -3.65722612 -4.19144066 -3.59909344]
Lars 의 정답률 : 
 [ 0.42265644  0.45491174 -0.62042565  0.38182299  0.53568101]
LarsCV 의 정답률 : 
 [0.50820001 0.35292847 0.5044123  0.42804021 0.48709115]
Lasso 의 정답률 :
 [0.28464234 0.24682018 0.27969503 0.33297802 0.35100836]
LassoCV 의 정답률 : 
 [0.50360677 0.57487532 0.38393926 0.49296728 0.22235781]
LassoLars 의 정답률 :
 [0.32740813 0.34916083 0.31900595 0.33201938 0.37731034]
LassoLarsCV 의 정답률 : 
 [0.31373431 0.53002649 0.550239   0.51720327 0.26800381]
LassoLarsIC 의 정답률 :
 [0.50953713 0.47459281 0.49722087 0.22791556 0.5231572 ]
LinearRegression 의 정답률 :
 [0.21005255 0.59145966 0.208458   0.53619264 0.52125793]
LinearSVR 의 정답률 :
 [-0.93195841 -0.34810868 -0.49269281 -0.40089365 -0.2672464 ]
MLPRegressor 의 정답률 : 
 [-3.36974763 -2.58668353 -3.03495983 -3.04119991 -3.20419674]
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
 [0.13495158 0.11321135 0.13862168 0.11666499 0.13611189]
OrthogonalMatchingPursuit 의 정답률 :
 [0.23774568 0.28409262 0.26173116 0.25714458 0.29797549]
OrthogonalMatchingPursuitCV 의 정답률 : 
 [0.38773449 0.48478831 0.520328   0.3746863  0.44196301]
PLSCanonical 의 정답률 :
 [-1.21423406 -1.21936835 -1.17887276 -1.44572078 -1.58897271]
PLSRegression 의 정답률 : 
 [0.397659   0.45868891 0.49293441 0.45992095 0.4824224 ]
PassiveAggressiveRegressor 의 정답률 :
 [0.43322606 0.36505021 0.40674628 0.53958259 0.3261684 ]
PoissonRegressor 의 정답률 : 
 [0.30168507 0.24014356 0.26808803 0.3269064  0.34960935]
RANSACRegressor 의 정답률 : 
 [ 0.06626398 -0.09625524  0.20144855 -1.92209434  0.08447886]
RadiusNeighborsRegressor 의 정답률 :
 [-2.65390103e-02 -9.15850639e-02 -2.66043537e-05 -3.29545929e-02
 -6.29418648e-03]
RandomForestRegressor 의 정답률 : 
 [0.39389905 0.39484837 0.34304419 0.36493773 0.21654857]
RegressorChain 은 없는놈!
Ridge 의 정답률 :
 [0.33674523 0.43517927 0.39013544 0.32451307 0.38724996]
RidgeCV 의 정답률 : 
 [0.3806704  0.5177871  0.47508269 0.47610569 0.41678027]
SGDRegressor 의 정답률 : 
 [0.40757146 0.32105421 0.3586766  0.3864718  0.36462487]
SVR 의 정답률 : 
 [0.07546279 0.06211986 0.09862553 0.10568178 0.12597923]
StackingRegressor 은 없는놈!
TheilSenRegressor 의 정답률 : 
 [0.39283226 0.44364376 0.28970256 0.45003998 0.56030329]
TransformedTargetRegressor 의 정답률 :
 [0.38721364 0.33300846 0.33414335 0.57444029 0.52185276]
TweedieRegressor 의 정답률 :
 [-0.00291228 -0.07583043 -0.00589793 -0.12167738 -0.01366437]
VotingRegressor 은 없는놈!
_SigmoidCalibration 의 정답률 : 
 [nan nan nan nan nan]
'''