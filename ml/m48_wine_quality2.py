import numpy as np
import pandas as pd

wine = pd.read_csv('../../data/csv/winequality-white.csv', sep = ';', header = 0)

count_data = wine.groupby('quality')['quality'].count()

print(count_data)

# print(np.unique(count_data['quality']))

import matplotlib.pyplot as plt
count_data.plot()
plt.show()