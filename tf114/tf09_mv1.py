import tensorflow as tf
tf.set_random_seed(66)

x1_data = [73. , 93., 89. , 96., 73.]
x2_data = [80. , 88., 91. , 98., 66.]
x3_data = [75. , 93., 90. , 100., 70.]
y_data = [152. , 185., 180. , 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name ='weight1')
w2 = tf.Variable(tf.random_normal([1]), name ='weight2')
w3 = tf.Variable(tf.random_normal([1]), name ='weight3')
b = tf.Variable(tf.random_normal([1]), name ='bias')

hypothesis = x1*w1 + x2 *w2 + x3 *w3 + b

cost = tf.reduce_mean(tf.square(hypothesis -y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00001) # larning_rate를 줄여봤다.
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001) :
    cost_val, hy_val, _ = sess.run([cost,hypothesis,train], 
                        feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    if step % 10 == 0 :
        print(step, "cost : ", cost_val, "\n",hy_val)
# .......
# 1960 cost :  9.939886 
#  [155.88474 181.61826 182.0535  197.22385 137.8218 ]
# 1970 cost :  9.887232
#  [155.87263 181.62657 182.0498  197.22113 137.83275]
# 1980 cost :  9.8348465
#  [155.86057 181.63486 182.04611 197.21838 137.84367]
# 1990 cost :  9.782724
#  [155.84851 181.64313 182.04243 197.21567 137.85457]
# 2000 cost :  9.730906
#  [155.83649 181.65137 182.03876 197.21295 137.86542]