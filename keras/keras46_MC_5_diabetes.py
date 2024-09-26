import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

x= dataset.data
y = dataset.target



print(x[:5])
print(y[:10])
print(x.shape, y.shape) #(442, 10) (442,)

print(np.max(x), np.min(y))
print(dataset.feature_names) # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(dataset.DESCR)




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, shuffle = True,
                                        random_state = 66)

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, train_size = 0.2)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout

input1 = Input(shape=(10,))
dense1 = Dense(70, activation= 'relu') (input1)
dense1 = Dense(50, activation= 'relu') (dense1)
# dense1 = Dropout(0.1) (dense1)
dense1 = Dense(35, activation= 'relu') (dense1)
dense1 = Dense(25, activation= 'relu') (dense1)
output1 = Dense(1) (dense1)

model = Model(input1, output1)

# compile, fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './modelCheckpoint/k46_diabetes_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='val_loss', patience= 5)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')

model.compile(loss = 'mse', optimizer= 'adam', metrics= ['acc'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size= 10, validation_data=(x_val,y_val),
                                        callbacks=[early_stopping,cp])

# evaluate, predict
loss = model.evaluate(x_test, y_test, batch_size=1, verbose =1)
print("loss : ", loss)
y_predict = model.predict(x_test)



'''
dropout 적용 전
loss :  [3292.3896484375, 47.62095260620117]
RMSE :  57.37935487178715
R2 :  0.4927010984337693

dropout 적용 후
loss :  [3687.185791015625, 50.94227981567383]
RMSE :  60.72220186124333
R2 :  0.4318701314278307
'''


import matplotlib.pyplot as plt

plt.figure(figsize= (10,6)) # 판을 깔아준다.

plt.subplot(2, 1, 1)    # 이미지 2개이상 섞겠다. (2,1)짜리 그림을 만들겠다. 2행2열중 1번째   
plt.plot(hist.history['loss'], marker='.', c='red',label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue',label='val_loss')
plt.grid()

plt.title('Cost loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)    # 이미지 2개이상 섞겠다. (2,1)짜리 그림을 만들겠다. 2행2열중 2번째   
plt.plot(hist.history['acc'], marker='.', c='red',label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue',label='val_acc')
plt.grid()

plt.title('Accuarcy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')


plt.show()