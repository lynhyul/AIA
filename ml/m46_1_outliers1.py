## 이상치 처리
## 1. 0 처리
## 2. 이상치라 예측되는 값을 NAN으로 바꿔 준 후에 보간법으로 채워본다.
## 3. 3,4,5......

import numpy as np

aaa = np.array([1,2,3,4,6,7,90,100,250,300])

def outliers(data_out) :
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25,50,75])
    print("1사 분위", quartile_1)
    print("q2 : ", q2)
    print("3사 분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out < lower_bound)) 

outlier_loc = outliers(aaa)
print("이상치의 위치 : ", outlier_loc)
 
# 1사 분위 3.25
# q2 :  6.5
# 3사 분위 :  97.5
# 이상치의 위치 :  (array([8, 9], dtype=int64),)

# Basic box plot
import matplotlib.pyplot as plt

plt.boxplot(aaa)
plt.show()

