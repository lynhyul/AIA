from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 진짜 맛있는 밥을 진짜 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index) # 횟수가 많은 순서대로, 횟수가 같을 경우에는 앞에 나온 단어부터 나열
# {'진짜': 1, '마구': 2, '나는': 3, '맛있는': 4, '밥을': 5, '먹었다': 6}

x = token.texts_to_sequences([text])
print(x)
# [[3, 1, 1, 4, 5, 1, 2, 2, 6]]

from keras.utils import to_categorical
word_size = len(token.word_index)
print(word_size)    # 6
x = to_categorical(x)

print(x)
'''
[[[0. 0. 0. 1. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 1. 0. 0.]
  [0. 0. 0. 0. 0. 1. 0.]
  [0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 1.]]]
'''
print(x.shape) # (1, 9, 7)
