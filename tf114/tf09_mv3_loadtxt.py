import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

dataset = np.loadtxt('../../data/csv/data-01-test-score.csv', delimiter=',')

x_data = dataset[:,:-1]
y_data = dataset[:,-1].reshape(25,1)
# print(dataset.shape) # (25, 4)
# print(x.shape) # (25, 4)
# print(y.shape) # (25, 1)


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

optimizer = tf.train.AdamOptimizer(learning_rate = 0.8) # larning_rate를 줄여봤다.
train = optimizer.minimize(cost)

# sess = tf.compat.v1.Session()

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for step in range(2001) :
        cost_val, hy_val, _ = sess.run([cost,hypothesis,train], 
                            feed_dict={x:x_data, y:y_data})
        if step % 100 == 0 :
            print(step, "cost : ", cost_val, "\n",hy_val)
    # predict / 갱신된 W와 b 로 hypothesis(model.predict)를 계산
    a = [[73,80,75],[93,88,93],[89,91,90],[96,98,100],[73,66,70]]
    print('예측결과 : ', sess.run(hypothesis, feed_dict = {x:a}))
# 예측결과 :  
# [[152.61342]
#  [185.07228]
#  [181.77515]
#  [199.73024]
#  [139.18759]]