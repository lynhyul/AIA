import numpy as np
import pandas as pd

result = []

for i in range(1,8) :
    x_pred = np.load(f"../../data/image/npy/x_pred{i}.npy",allow_pickle=True)
    result.append(x_pred)
    # print(result.shape)
result = np.mean(result, axis=0)
sub = pd.read_csv('../../data/image/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('../../data/image/tta_answer1.csv',index=False)
