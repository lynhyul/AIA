import matplotlib.pyplot as plt
import tensorflow as tf

x = [1. , 2. , 3.]
y = [3. , 5. , 7.]

w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w

cost = tf.reduce_mean(tf.square(hypothesis-y))


w_hsitory = []
cost_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50) :
        sess.run(tf.compat.v1.global_variables_initializer())
        curr_w = i* 0.1     # -3~4.9까지 0.1단위로 증가
        curr_cost = sess.run(cost, feed_dict = {w:curr_w})

        w_hsitory.append(curr_w)
        cost_history.append(curr_cost)

print("=========================================")
print(w_hsitory)
print("=========================================")
print(cost_history)
print("=========================================")

plt.plot(w_hsitory, cost_history)
plt.show()