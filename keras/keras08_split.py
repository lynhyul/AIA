from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

# 1. Data
x = np.array(range(1,101))      # 1 ~ 100   # x = np.array(range(100))        # 0 ~ 99
y = np.array(range(101,201))    # 101 ~ 200

x_train = x[:60]                # 1 ~ 60
x_val = x[60:80]                # 61 ~ 80
x_test = x[80:]                 # 81 ~ 100

y_train = y[:60]                # 1 ~ 60
y_val = y[60:80]                # 61 ~ 80
y_test = y[80:]                 # 81 ~ 100


# 2. model
model = Sequential()
model.add(Dense(10, input_dim = 1, activation = "relu"))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))

# 3. compile and traning
model.compile(loss = "mse", optimizer = "adam", metrics = ["mae"])
model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_data = (x_val, y_val)) 

# 4. 예측 및 평가
results = model.evaluate(x_test, y_test, batch_size = 1)
print("results(mes, mae): ", results)

y_predict = model.predict(x_test)
# print("y_predict: ", "\n", y_predict)

# 사이킷런 (sklearn)
from sklearn.metrics import mean_squared_error

def RMSE(y_test , y_predict): # y 예측값과 실제 y갑의 RMSE값
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("mean_squared_error: ", mean_squared_error(y_predict, y_test))
print("RMSE: ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score # R2(결정계수)
r2 = r2_score(y_test, y_predict)
print("R2: ", r2)