#!/usr/bin/python
# -*- coding: utf-8 -*-
# EX05 以下是一个完整的简单的神经网络的程序
import tensorflow as tf
from numpy.random import RandomState
from basic_python.basictools import pause


def print_tensor(tensor):
    '''
    打印一个张量
    :param tensor:要打印的张量
    :return: None
    '''
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)  # 在此利用以上两行对其中涉及的参数进行统一初始化
        print(sess.run(tensor))


batch_size = 8  # 定义训练数据batch的大小
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=2))  # 初试化权重矩阵（生成相应大小的两个随机正态分布，均值为0，方差为1）

# LGM B
print(print_tensor(w1))
pause(True)
print(print_tensor(w2))
pause(True)
# LGM E

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')  # 在第一个维度上利用None，可以方便使用不大的batch大小
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')  # 在第一个维度上利用None，可以方便使用不大的batch大小

# 定义前向传播过程
a = tf.matmul(x, w1)  # 计算隐层的值
y = tf.matmul(a, w2)  # 计算输出值

# 定义损失函数和反向传播算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)  # 在此神经网络中只有基础学习率的设置，没有指数衰减率，也没有正则项和滑动平均模型。
# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 以下定义真实的样本标签（利用如下规则：所有x1+x2<1的样例被认为正样本，即合格样本，用1表示；否则不合格样本，用0表示）
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]
# 以下创建一个session会话来运行TensorFlow程序
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)  # 在此利用以上两行对其中涉及的参数进行统一初始化
    print(sess.run(w1))
    print(sess.run(w2))  # 在此将会打印出训练神经网络前的参数值
    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)  # 每次选取batch_size个样本进行训练
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})  # 通过选取的样本训练神经网络并更新其中的参数
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After%dtraining step(s),cross_entropy on all data is%g" % (i, total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))