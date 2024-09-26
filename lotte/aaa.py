# import pandas as pd

# aaa =pd.DataFrame(index = range(10))
# aaa.loc[0,:] = [1,2]


# print(aaa)

import numpy as np
import pandas as pd

result2 = []

result = np.arange(0,1,0.01)
result = result.reshape(10,10)  # 10,10


test_img_idx = 0
index_record = []
per_idx_record = []
percent_record = []
for test_img_percent in result:
    percent_idx = 0
    for percent in test_img_percent:
        if percent > 0.1 and percent < 0.5:
            index_record.append(test_img_idx)
            percent_record.append(percent)
            per_idx_record.append(percent_idx)
        percent_idx += 1
    test_img_idx += 1

df = pd.DataFrame({'test_index':index_record, 'answer': per_idx_record, 'percent': percent_record}, index = None)
df.to_csv('../../data/image/just_test.csv',index=False)
print(df)