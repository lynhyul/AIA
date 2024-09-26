# 회귀 모델
# 최종 sklearn의 R2값으로 결론낼것!!

from sklearn.datasets import load_diabetes
import tensorflow as tf

tf.set_random_seed(66)

datasets = load_diabetes()

x_data = datasets.data 
y_data = datasets.target

from sklearn.model_selection import train_test_split

y_data = y_data.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size = 0.2, random_state=66)



# print(x_data.shape) # 442,10

x = tf.placeholder(tf.float32, shape = (None,10))
y = tf.placeholder(tf.float32, shape = (None,1))

w = tf.Variable(tf.random_normal([10,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.matmul(x,w) + b

cost = tf.reduce_mean(tf.square(hypothesis -y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.8) # larning_rate를 줄여봤다.
train = optimizer.minimize(cost)

from sklearn.metrics import r2_score
with tf.Session() as sess : #with문을쓰면 자동으로 close역할도 수행한다.
    sess.run(tf.compat.v1.global_variables_initializer()) #변수 초기화
    for step in range(10001) :
        cost_val, hy_val, _ = sess.run([cost,hypothesis,train], 
                            feed_dict={x:x_train, y:y_train})
        if step % 1000 == 0 :
            print(step, "cost : ", cost_val)

    predict = sess.run(hypothesis, feed_dict = {x:x_test})
    r2 = r2_score(y_test,predict)
    print("R2 : ",r2)

# 5000 cost :  2801.9211
# R2 :  0.5063891162769933

