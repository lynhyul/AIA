import tensorflow as tf
print(tf.__version__)   # 1.14.0

hello = tf.constant("Hello World")

print(hello)
# Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()
print(sess.run(hello))
# b'Hello World'