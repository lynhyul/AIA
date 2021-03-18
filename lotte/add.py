import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

# for i in range(1,6) :
#     df1.add(globals()['df{}'.format(i)], axis=1)
# df = df1.iloc[:,1:]
# df_2 = df1.iloc[:,:1]
# df_3 = (df/5).round(2)
# df_3.insert(0,'id',df_2)
# df3.to_csv('../data/csv/0122_timeseries_scale10.csv', index = False)

x = []
for i in range(4,8):
    df = pd.read_csv(f'../../data/image/answer{i}.csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)

df = pd.read_csv(f'../../data/image/answer{i}.csv', index_col=0, header=0)
for i in range(72000):
    for j in range(1):
        a = []
        for k in range(4):
            a.append(x[k,i,j].astype('float32'))
        a = np.array(a)
        df.iloc[[i],[j]] = (pd.DataFrame(a).astype('float32').quantile(0.5,axis = 0)[0]).astype('float32')

y = pd.DataFrame(df, index = None, columns = None)
y = y.to_numpy()
y = np.round(y, 0)
# print(y)
sub = pd.read_csv('../../data/image/sample.csv')
sub['prediction'] = pd.DataFrame(y)
sub.to_csv('../../data/image/answer_add_s3.csv',index=False)