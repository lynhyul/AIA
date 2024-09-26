import tensorflow as tf
tf.set_random_seed(66)  # 이걸 사용하지 않으면 돌릴 때 마다 값이 달라진다.

x_train = [1,2,3]
y_train = [3,5,7]
# W = tf.Variable(tf.random_normal([1]), name = 'weight')
# # weight값 정규분포에 의한 랜덤한 값 한 개를 집어 넣는다.
# b = tf.Variable(tf.random_normal([1]), name = 'bias')

W = tf.Variable(0.9, name = 'weight') # 정규분포에 의한 랜덤한 값 한 개를 집어 넣는다.
b = tf.Variable(0.3, name = 'bias')

# sess = tf.Session()
# sess.run(tf.compat.v1.global_variables_initializer()) #변수 초기화
# print(sess.run(W), sess.run(b)) # [0.06524777] [1.4264158]

hypothesis = x_train * W + b # y=wx+b의 형태

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # loss = mse

optimizer =tf.train.GradientDescentOptimizer(learning_rate=0.01) #옵티마이저
train = optimizer.minimize(cost)

# sess 선언 및 변수 초기화
sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer()) #변수 초기화

for step in range(100) :                   # 2001번을 돌리겠다.
    print(step,sess.run(cost), sess.run(W), sess.run(b))
    sess.run(train)
    # if step %  ==0:
    

# 0 11.376854 [0.22876799] [1.4952775]
# 1 9.026286 [0.37427187] [1.5562212]
# 2 7.1681275 [0.50375766] [1.6101259]
# 3 5.699192 [0.6190019] [1.657773]
# 4 4.5379176 [0.7215842] [1.6998575]
# 5 3.6198337 [0.8129087] [1.736997]    
'''
0 11.376854 [0.22876799] [1.4952775]
20 0.25022563 [1.4284163] [1.9631107]
40 0.13584478 [1.561582] [1.9646112]
60 0.12254739 [1.5923083] [1.9237307]
80 0.11129215 [1.612432] [1.8807428]
100 0.10107721 [1.6307381] [1.8393915]
....
1920 1.5844156e-05 [1.9953768] [1.0105095]
1940 1.4392031e-05 [1.9955939] [1.010016]
1960 1.3069816e-05 [1.9958011] [1.0095452]
1980 1.187079e-05 [1.9959984] [1.0090965]
2000 1.0781078e-05 [1.9961864] [1.0086691]
'''
