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
for i in range(1,21):
    if i != 10:
        df = pd.read_csv(f'../../data/image/add/answer ({i}).csv', index_col=0, header=0)
        data = df.to_numpy()
        x.append(data)

x = np.array(x)

print(x.shape)


a= []
c = []
df = pd.read_csv(f'../../data/image/answer ({i}).csv', index_col=0, header=0)

for i in range(72000):
        b = []
        
        for k in range(19):
            b.append(x[k,i,0].astype('int'))
        
        if 4 > stats.mode(b)[1]:
            a.append(stats.mode(b)[0]) 
            c.append(i)
       

c = np.array(c)
print(c.shape)
a = np.array(a)
print(a.shape)

print(c)

# a = a.reshape(72000,4)

sub = pd.read_csv('../../data/image/sample1.csv')
sub['prediction'] = np.array(c)
sub.to_csv('../../data/image/sample3.csv',index=False)


# 24 => 86점 / 25 => 86.229 / 26 => 86.394 / 27 => 86.461 / 28 => 86.492 / 29 => 86.586 / 35=> 88.419 / 36 => 88.425
