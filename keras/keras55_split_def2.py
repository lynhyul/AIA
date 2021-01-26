import numpy as np
import pandas as pd

data = np.load('c:/data/npy/삼성전자2.npy',allow_pickle=True)[0].reshape(659,24)
print(data.shape) # (659,24)


def split_xy(dataset, timesteps_x, timesteps_y, x_column, y_column):
    x, y = list(), list()
    
    for i in range(len(data)):
        x_end_number = i + timesteps_x
        y_end_number = x_end_number + timesteps_y
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number : y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    x = np.array(x)
    y = np.array(y)
    print(x.shape, y.shape) # 653,5, 24 / 653, 2, 24
    x = x[:,:,:x_column]
    y = y[:,:,:y_column]
    return x, y

timesteps_x = 5
timesteps_y =2 
x_column = 13
y_column = 5
x, y = split_xy(data, timesteps_x, timesteps_y, x_column, y_column)
print(x.shape, y.shape) # (653, 5, 13) (653, 2, 5)
print(x)