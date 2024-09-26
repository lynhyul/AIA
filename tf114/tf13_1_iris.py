import tensorflow as tf
from sklearn.datasets import load_iris
import numpy as np
tf.set_random_seed(66)

datasets = load_iris() 

x_data = datasets.data
y_data = datasets.target.reshape(-1,1)

print(x_data.shape) # 150,4


x = tf.placeholder(tf.float32, shape = [None,4])
y = tf.placeholder(tf.float32, shape = [None,3])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
encoder = OneHotEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()

# 여기서 훈련 데이터만 원핫인코딩을 해주는 이유가있는데
# 일단, 사이킷런에서 제공하는 accuarcy_score는 1차원 배열밖에 받지 못한다.
# 쉽게 말해 [[0. 1. 0.], [0. 1. 0.]...] 이런 형태를 받지 못한다. 
# 그래서 해결방법이 2가지가 있더라.
# 1번째로, 위 처럼 train데이터에 대해서만 원핫 인코딩을 해준다(일단 훈련데이터를 인코딩 해줘야만 돌아가더라)
# 2번째로, y_data에 대해서 인코딩 후에 훈련되서 나온 가중치와 x_test를 predict한 값과 y_test값을 둘 다
# np.argmax를 이용하여 배열을 통일 시켜줘서 나오게 하는 방법이 있다. 

w = tf.Variable(tf.zeros([4,3], name='weight'))
b = tf.Variable(tf.zeros([1,3]), name = 'bias') # y의 출력의 갯수에 맞게 조정한다.(행렬의 덧셈 방법 참고)

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

# cost = tf.reduce_mean(tf.square(hypothesis -y))
loss = tf.reduce_mean(-tf.reduce_sum(y* tf.log(hypothesis), axis=1)) # categorical_crossentropy
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

from sklearn.metrics import accuracy_score

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) # 변수 초기화

    for step in range(2001) :
        loss_val, _ = sess.run([loss,train], feed_dict = {x:x_train, y:y_train})

        if step % 200 ==0 :
            print("epochs : ", step, "\n loss : ", loss_val)

    # predict
    y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis= 1)
    print('accuracy_score : ', accuracy_score(y_test, y_pred))
    # accuracy_score :  1.0
    
