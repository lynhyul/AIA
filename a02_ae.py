import numpy as np
from tensorflow.keras.datasets import mnist

# 오토인코더 - 비지도 학습, 차원축소에도 사용
# y값이 없다!!
# 784, 엠니스트 데이터가 64 덴스레이어로 들어가고 다시 784, 로 나온다면
# 데이터 축소, 데이터 확장이 같이 이뤄진다

(x_train, _), (x_test, _) = mnist.load_data() # _ 은 쓰지 않을 변수, y 값을 사용하지 않을것이다!

x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784)/255.

# print(x_train[0]) # 잘출력되는지 확인
# print(x_test[0])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units = hidden_layer_size, activation = 'relu', input_shape = (784,)))
    model.add(Dense(units = 784, activation = 'sigmoid'))
    return model

model = autoencoder(hidden_layer_size= 154)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

model.fit(x_train, x_train, epochs = 10)

output = model.predict(x_test)

# 시각화! 이번에는 테스트 데이터중 랜덤한 5개를 가져온다!
# 이전 코드와 달라진점은 중간 레이어 154개!
# PCA 개념과 유사하다 (PCA 도 오토인코더의 일부분...?)
# PCA 에서 64차원으로 줄이면 80퍼센트, 154차원으로 줄이면 95퍼센트를 보여준걸 생각하면 된다!!
## 덴스레이어의 노드가 늘어날수록 복원도도 높아진다!


from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize = (20, 7))

# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


plt.tight_layout()
plt.show()