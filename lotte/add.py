import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from scipy import stats

x = []
for i in range(1,36):           # 파일의 갯수
    df = pd.read_csv(f'../../data/image/add/answer ({i}).csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)
                

x = np.array(x)

# print(x.shape)
a= []
df = pd.read_csv(f'../../data/image/add/answer ({i}).csv', index_col=0, header=0)
for i in range(72000):
    for j in range(1):
        b = []
        for k in range(35):         # 파일의 갯수
            b.append(x[k,i,j].astype('int'))
        a.append(stats.mode(b)[0]) 
# a = np.array(a)
# a = a.reshape(72000,4)

# print(a)

sub = pd.read_csv('../../data/image/sample.csv')
sub['prediction'] = np.array(a)
sub.to_csv('../../data/image/answer_add/answer_add_s47.csv',index=False)

# 24 => 86점 / 25 => 86.229 / 26 => 86.394 / 27 => 86.461 / 28 => 86.492 / 29 => 86.586 / 35=> 88.419 / 36 => 88.425

#최종제출 40_7 / 41