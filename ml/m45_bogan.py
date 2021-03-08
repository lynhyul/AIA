from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datestrs = ['3/1/2021', '3/2/2021', '3/3/2021', '3/4/2021', '3/5/2021']
dates = pd.to_datetime(datestrs)
print(dates)
print("================================")

ts = Series([1, np.nan, np.nan, 8, 10], index=dates)
print(ts)
# 2021-03-01     1.0
# 2021-03-02     NaN
# 2021-03-03     NaN
# 2021-03-04     8.0
# 2021-03-05    10.0

ts_intp_linear = ts.interpolate()
print(ts_intp_linear)       # pandas 모듈로 보간법을 이용해서 nan값을 채워나간다. / 가급적이면 시계열 데이터에서 사용 할 것
# 2021-03-01     1.000000
# 2021-03-02     3.333333
# 2021-03-03     5.666667
# 2021-03-04     8.000000
# 2021-03-05    10.000000