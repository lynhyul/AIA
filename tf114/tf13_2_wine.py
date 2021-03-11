import tensorflow as tf
from sklearn.datasets import load_wine
import numpy as np
tf.set_random_seed(66)

datasets = load_wine() 

x_data = datasets.data
y_data = datasets.target.reshape(-1,1)

print(x_data.shape) # 178,13




x = tf.placeholder(tf.float32, shape = [None,13])
y = tf.placeholder(tf.float32, shape = [None,3])



from sklearn.preprocessing import OneHotEncoder, LabelEncoder
encoder = OneHotEncoder()
encoder.fit(y_data)
y_data = encoder.transform(y_data).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=56)


# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# encoder = OneHotEncoder()
# encoder.fit(y_train)
# y_train = encoder.transform(y_train).toarray()

print("y_test(no-argmax) : ",y_test)
#  ...
#  [0. 1. 0.]
#  [0. 1. 0.]
#  ...
print("y_test(argmax) : ",np.argmax(y_test,1)) 
# y_test :  [2 1 1 0 1 1 2 0 0 1 2 0 1 1 1 2 2 0 1 2 1 0 0 0 0 0 1 1 0 1 1 0 2 0 1 0 2 2 1 0]
# y_pred :  [1 0 1 0 1 1 0 0 0 1 0 0 0 1 1 1 1 0 1 1 1 0 0 0 0 0 0 1 0 1 1 0 1 0 1 1 1 1 1 0]

w = tf.Variable(tf.zeros([13,3], name='weight'))
b = tf.Variable(tf.zeros([1,3]), name = 'bias') # y의 출력의 갯수에 맞게 조정한다.(행렬의 덧셈 방법 참고)

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

# cost = tf.reduce_mean(tf.square(hypothesis -y))
loss = tf.reduce_mean(-tf.reduce_sum(y* tf.log(hypothesis), axis=1)) # categorical_crossentropy
train = tf.train.GradientDescentOptimizer(learning_rate=0.000002).minimize(loss)

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
    y_test = np.argmax(y_test, axis =1)
    print("y_pred : ",y_pred)
    print("y_test : ", y_test)
    print('accuracy_score : ', accuracy_score(y_test, y_pred))
    # accuracy_score :  0.7222222222222222