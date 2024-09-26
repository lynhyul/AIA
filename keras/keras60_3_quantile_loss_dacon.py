import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow.keras.backend as K
# import tensorflow.keras.layers as L
# import tensorflow.keras.models as M


# def custom_mse(y_true, y_pred) :
#     return tf.math.reduce_mean(tf.square(y_true-y_pred))


# def quantile_loss(y_true,y_pred) :
#     qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ,0.9]
#     q = tf.constant(np.array([qs]), dtype=tf.float32)   # 
#     e = y_true - y_pred
#     v = tf.maximum(q*e, (q-1)*e)
#     return k.mean(v)


def quantile_loss_dacon(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]



x = np.array([1,2,3,4,5,6,7,8]).astype('float32')
y = np.array([1,2,3,4,5,6,7,8]).astype('float32')

print(x.shape) # 8,1


model = Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(1))

# model.compile(loss =custom_mse, optimizer= 'adam')
# model.compile(loss = 'mse', optimizer= 'adam')
for q in quantiles :
    model.compile(loss = lambda y_true, y_pred: quantile_loss_dacon(q, y_true, y_pred), optimizer='adam')
    model.fit(x,y, batch_size=1, epochs=1)

loss = model.evaluate(x,y)

print(loss)
'''
model.compile(loss =custom_mse, optimizer= 'adam')
54.853309631347656
model.compile(loss =quantile_loss, optimizer= 'adam')
2.9665937423706055
'''