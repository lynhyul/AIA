# 실습
# outliers1 을 행렬형태도 적용할 수 있도록 수정
# 혜지님 감사합니다~

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],
               [1000,2000,3,4000,5000,-100000,7000,8,9000,10000]])
aaa = aaa.transpose()
print(aaa.shape) # (10, 2)


def outliers(data_out):
    allout = []
    for i in range(data_out.shape[1]):
        quartile_1, q2, quartile_3 = np.percentile(data_out[:,i], [25, 50, 75])
        print('1사분위(25%지점): ',  quartile_1)
        print('q2(50%지점): ',  q2)
        print('3사분위(75%지점): ',  quartile_3)
        iqr = quartile_3 - quartile_1   # IQR(InterQuartile Range, 사분범위)
        print('iqr: ', iqr)
        lower_bound = quartile_1 - (iqr * 1.5)  # 하계
        upper_bound = quartile_3 + (iqr * 1.5)  # 상계
        print('lower_bound: ', lower_bound)
        print('upper_bound: ', upper_bound)

        a = np.where((data_out[:,i]>upper_bound) | (data_out[:,i]<lower_bound)) 
        allout.append(a)

    return np.array(allout)


outlier_loc = outliers(aaa)
print('이상치의 위치: ', outlier_loc)

# 1사분위(25%지점):  3.25
# q2(50%지점):  6.5
# 3사분위(75%지점):  97.5
# iqr:  94.25
# lower_bound:  -138.125
# upper_bound:  238.875
# 1사분위(25%지점):  256.0
# q2(50%지점):  3000.0
# 3사분위(75%지점):  6500.0
# iqr:  6244.0
# lower_bound:  -9110.0
# upper_bound:  15866.0
# 이상치의 위치:  [[array([4, 7], dtype=int64)]
#  [array([5], dtype=int64)]]