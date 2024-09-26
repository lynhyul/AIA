from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요','참 최고에요','참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요','글쎄요',
        '별로에요','생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요','규현이가 잘 생기긴 했어요']

# 긍정1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 6, '영화에요': 7, '추천하고': 8, 
# '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, 
# '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, 
# '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '규현이가': 25, '생기긴': 26, '했어요': 27}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], 
# [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]


from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='post') #post, 길이를 맞춰주는 함수
print(pad_x)
'''
[[ 2  4  0  0  0]
 [ 1  5  0  0  0]
 [ 1  3  6  7  0]
 [ 8  9 10  0  0]
 [11 12 13 14 15]
 [16  0  0  0  0]
 [17  0  0  0  0]
 [18 19  0  0  0]
 [20 21  0  0  0]
 [22  0  0  0  0]
 [ 2 23  0  0  0]
 [ 1 24  0  0  0]
 [25  3 26 27  0]]
 '''
print(pad_x.shape) # (13, 5)

## 위의 (13,5)를 (13,4)가 되게 해보렴

# pad_x = pad_sequences(x, maxlen=4,padding='post') #post, 길이를 맞춰주는 함수
# print(pad_x)
# print(pad_x.shape) # (13, 4)
# '''
# [[ 2  4  0  0]
#  [ 1  5  0  0]
#  [ 1  3  6  7]
#  [ 8  9 10  0]
#  [12 13 14 15]
#  [16  0  0  0]
#  [17  0  0  0]
#  [18 19  0  0]
#  [20 21  0  0]
#  [22  0  0  0]
#  [ 2 23  0  0]
#  [ 1 24  0  0]
#  [25  3 26 27]]
# '''
# print(np.unique(pad_x))
# # [ 0  1  2  3  4  5  6  7  8  9 10 12 13 14 15 16 17 18 19 20 21 22 23 24
# #  25 26 27]
# print(len(np.unique(pad_x))) # 27 / 0부터 27까지인데, 11이 maxlen=4로 인해 잘렸다.

print(len(np.unique(pad_x))) # 28 / maxlen 적용x


from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

model = Sequential()
model.add(Embedding(input_dim=28, output_dim=11, input_length= 5)) # 아웃풋은 임의로 넣어줘도 상관은 없다.
                                                                   # input_length는 13,5중에서 5가 들어간다.
# model.add(Flatten())
model.add(Conv1D(32, 2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 11)             308
_________________________________________________________________
conv1d (Conv1D)              (None, 4, 32)             736
_________________________________________________________________
flatten (Flatten)            (None, 128)               0
_________________________________________________________________
dense (Dense)                (None, 1)                 129
=================================================================
Total params: 1,173
Trainable params: 1,173
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics=['acc'])
# 옵티박께서 말씀하시길.... lstm한정 sgd, 갓담보다 성능이 좋게 나오기도 한다고 합니다.
model.fit(pad_x, labels, epochs= 100)

acc = model.evaluate(pad_x, labels) [1]
# loss: 0.0050 - acc: 1.0000