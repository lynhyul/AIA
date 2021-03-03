from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = reuters.load_data(
        num_words= 1000, test_split= 0.2
)

print(x_train[0], type(x_train[0])) # <class 'list'>

print(y_train[0]) # 3
print("===========================================")
print(x_train.shape, x_test.shape) # (8982,) (2246,)
print(y_train.shape, y_test.shape) # (8982,) (2246,)


print(x_train.shape, x_test.shape) # (8982, 100) (2246, 100)
print(y_train.shape, y_test.shape) # (8982, 46) (2246, 46)

# print("뉴스기사 최대길이 : ", max(len(l)for l in x_train)) # 뉴스기사 최대길이 :  2376
# print("뉴스기사 평균길이 : ", sum(map(len,x_train)) / len(x_train)) # 뉴스기사 평균길이 :  145.5398574927633


# plt.hist([len(s) for s in x_train], bins= 50)
# plt.show()

# # y quntaile
# unique_elements, counts_elements = np.unique(y_train,return_counts= True)
# print("y_quntaile :", dict(zip(unique_elements, counts_elements)))
# '''
# y_quntaile : {0: 55, 1: 432, 2: 74, 3: 3159, 4: 1949, 5: 17, 6: 48, 7: 16, 8: 139, 9: 101, 10: 124, 11: 390, 
#  12: 49, 13: 172, 14: 26, 15: 20, 16: 444, 17: 39, 18: 66, 19: 549, 20: 269, 21: 100, 22: 15,
#  23: 41, 24: 62, 25: 92, 26: 24, 27: 15, 28: 48, 29: 19, 30: 45, 31: 39, 32: 32, 33: 11, 34: 50, 35: 10, 36: 49, 
#  37: 19, 38: 19, 39: 24, 40: 36, 41: 30, 42: 13, 43: 21, 44: 12, 45: 18}
# '''

# print("=========================================")

# plt.hist(y_train, bins= 46)
# plt.show()

# # x_word quntaile
# word_to_index = reuters.get_word_index()
# print(word_to_index)
# '''
# {'mdbl': 10996, 'fawc': 16260, 'degussa': 12089, 'woods': 8803, 'hanging': 13796, 'localized': 20672, 'sation': 20673, 
# 'chanthaburi': 20675, 'refunding': 10997, 'hermann': 8804, 'passsengers': 20676, 'stipulate': 20677, 'heublein': 8352, 
# 'screaming': 20713, 'tcby': 16261, 'four': 185, 'grains': 1642, 'broiler': 20680,
#  'wooden': 12090, 'wednesday': 1220, 'highveld': 13797, 'duffour': 7593, '0053': 20681, 'elections': 3914
#  ... paradyne's": 20814, '691': 6363, 'paychecks': ~ }
#  '''
# print(type(word_to_index)) # <class 'dict'>
# print("======================================")

# # key, value
# index_to_word = {}
# for key, value in word_to_index.items() :
#     index_to_word[value] = key

# print(index_to_word)
# # {10996: 'mdbl', 16260: 'fawc', 12089: 'degussa', 8803: 'woods', 13796: 'hanging', ...}
# print(index_to_word[1]) # the
# print(index_to_word[30979]) # northerly
# print(len(index_to_word)) # 30979

# # x_train[0]
# print(x_train[0])
# '''
# [1, 27595, 28842, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102,
#  6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 
#  155, 11, 15, 7,48, 9, 4579, 1005, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 1245, 
#  90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12]
# '''
# print(' '.join(index_to_word[index] for index in x_train[0]))
# '''
# the wattie nondiscriminatory mln loss for plc said at only ended said commonwealth could 1 traders now april 0 
# a after said from 1985 and from foreign 000 april 0 prices its account year a but in this mln home an states 
# earlier and rise and revs vs 000 its 16 vs 000 a but 3 psbr oils several and 
# shareholders and dividend vs 000 its all 4 vs 000 1 mln agreed largely april 0 are 2 states will 
# billion total and against 000 pct dlrs
# '''

# y category
category = np.max(y_train) + 1
print("y_category : ", category) # 46

# y unique
y_bunpo = np.unique(y_train)
print(y_bunpo)


from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 550
x_train = pad_sequences(x_train, maxlen=max_len) # 훈련용 뉴스 기사 패딩
x_test = pad_sequences(x_test, maxlen=max_len) # 테스트용 뉴스 기사 패딩

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

from keras.models import Sequential, load_model
from keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
model = Sequential()
model.add(Embedding(1000, 256, input_length = max_len))
model.add(LSTM(256))
model.add(Dropout(0.4))
model.add(Dense(46, activation='softmax'))
model.summary()



from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=30)
mc = ModelCheckpoint('../data/modelcheckpoint/best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
lr = ReduceLROnPlateau(factor=0.5, verbose=1, patience=15)

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(1e-3), metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=128, epochs=100, callbacks=[es, mc,lr],
 validation_data=(x_test, y_test)
)
# loaded_model = load_model('../data/modelcheckpoint/best_model.h5')
print("\n 테스트 정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))

# 테스트 정확도: 0.7858

# epochs = range(1, len(history.history['acc']) + 1)
# plt.plot(epochs, history.history['loss'])
# plt.plot(epochs, history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()