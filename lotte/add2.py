import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from scipy import stats

# for i in range(1,6) :
#     df1.add(globals()['df{}'.format(i)], axis=1)
# df = df1.iloc[:,1:]
# df_2 = df1.iloc[:,:1]
# df_3 = (df/5).round(2)
# df_3.insert(0,'id',df_2)
# df3.to_csv('../data/csv/0122_timeseries_scale10.csv', index = False)

x = []
for i in range(1,9):
    if i != 2:
        if i != 3:
            if i != 6:
                if i != 8:
                    df = pd.read_csv(f'../../data/image/add2/answer ({i}).csv', index_col=0, header=0)
                    data = df.to_numpy()
                    x.append(data)

x = np.array(x)

# print(x.shape)
a= []
df = pd.read_csv(f'../../data/image/add2/answer ({i}).csv', index_col=0, header=0)
for i in range(72000):
    for j in range(1):
        b = []
        for k in range(4):
            b.append(x[k,i,j].astype('int'))
        a.append(stats.mode(b)[0]) 
# a = np.array(a)
# a = a.reshape(72000,4)

# print(a)

sub = pd.read_csv('../../data/image/sample.csv')
sub['prediction'] = np.array(a)
sub.to_csv('../../data/image/answer_add/answer_add_s.csv',index=False)