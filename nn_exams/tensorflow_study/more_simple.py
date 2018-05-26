# -*- coding: utf-8 -*-

# https://www.cnblogs.com/hypnus-ly/p/8047214.html
# 自定义损失函数的神经网络
import tensorflow as tf
from numpy.random import RandomState

batch_size=8

# 定义两个输入节点
x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')
# 回归问题一般只有一个输出节点
y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')

# 定义一个单层的神经网络
w1=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y=tf.matmul(x,w1)

# 使用上述自己定义的损失函数
a=1;b=10
loss=tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*a,(y_-y)*b))

train_step=tf.train.AdamOptimizer(0.001).minimize(loss)

# 随机生成一个模拟数据集
rdm=RandomState(1)
dataset_size=128
X=rdm.rand(dataset_size,2)
# 设置随机噪声,范围在-0.05~0.05
Y = [[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]

# 训练神经网路
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS=5000
    for i in range(STEPS):
        start=(i*batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        print(sess.run(w1))