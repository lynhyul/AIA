import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

# aaa =np.array([[73,51,65],[92,98,11],[89,31,33],[99,33,100],[17,66,79]])  # 5,3
# print(aaa.shape)

x_data = [[73,51,65],[92,98,11],[89,31,33],[99,33,100],[17,66,79]] # 5,3
y_data = [[152],[185],[180],[205],[142]]    # 5,1

x = tf.placeholder(tf.float32, shape = [None,3])
y = tf.placeholder(tf.float32, shape = [None,1])

w = tf.Variable(tf.random_normal([3,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# >> 1,3 형태로 한개의 x 데이터가 들어감
# >> hypo 만들어줌
# >> hypo - y(한개)   >> 얘를 데이터수만큼 반복
# >> 오차함수 계산   >> 데이터수만큼 반복한걸 평균내줌


# hypothesis = x * w+b -> 오류가 뜬다.
hypothesis = tf.matmul(x,w) + b # matmul => 행렬 전용 곱셈

# [실습]
# verbose로 나오는 놈은  step과 cost와 hypothesis를 출력

cost = tf.reduce_mean(tf.square(hypothesis -y))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.9) # larning_rate를 줄여봤다.
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001) :
    cost_val, hy_val, _ = sess.run([cost,hypothesis,train], 
                        feed_dict={x:x_data, y:y_data})
    if step % 100 == 0 :
        print(step, "cost : ", cost_val, "\n",hy_val)

        

