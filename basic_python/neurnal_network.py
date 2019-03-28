#!/usr/bin/python
# -*- coding: utf-8 -*-
# https://blog.csdn.net/zhouhailin1007/article/details/84102028
# "pd"偏导，这是个最原始的利用numpy实现的三层网络，参照上述文件，自行修改版本，和利用向量表示的版本（简化）完全对照起来

import numpy as np
import torch

alpha = 0.5  # 学习率
num_iter = 2  # 迭代次数

def sigmoid(x):
    """
    激活函数
    """
    return 1 / (1 + np.exp(-x))


def print_(inf, x):
    """
    计算激活函数的偏微分
    """
    print(inf, "的值为：")
    print("---------------------")
    print(x)
    print("---------------------")
    print("以上是", inf, "的值")
    print()
    return


def sigmoid_derivation(y):
    """
    求解sigmoid函数的导数
    :param y: 因变量
    :return: 导数
    """
    return y * (1 - y)


def create_by_classic():
    """
    这是通过传统方法，或者说是最原始的方法实现的三层神经网络，激活函数用的sigmoid，算是最经典的方法了，网络的由来参考
    https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    :return: 
    """
    # 初始化
    bias = [0.35, 0.60]
    weight = [0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.55]
    # 初始值
    i1 = 0.05
    i2 = 0.10
    # 目标值
    target1 = 0.01
    target2 = 0.99

    for i in range(num_iter):
        # 正向传播
        net_h1 = i1 * weight[1 - 1] + i2 * weight[2 - 1] + bias[0]
        net_h2 = i1 * weight[3 - 1] + i2 * weight[4 - 1] + bias[0]
        out_h1 = sigmoid(net_h1)
        out_h2 = sigmoid(net_h2)
        net_o1 = out_h1 * weight[5 - 1] + out_h2 * weight[6 - 1] + bias[1]
        net_o2 = out_h1 * weight[7 - 1] + out_h2 * weight[8 - 1] + bias[1]
        out_o1 = sigmoid(net_o1)
        out_o2 = sigmoid(net_o2)
        # 输出迭代后的预测值
        print(str(i) + "：result: " + str(out_o1) + ",result: " + str(out_o2))
        # 输出误差，可以通过改变迭代次数来查看效果
        print(str(i) + "：error: " + str(0.5 * (np.square(target1 - out_o1) + np.square(target2 - out_o2))))

        if i == num_iter - 1:
            print("latest result ：" + str(out_o1) + ",result: " + str(out_o2))
            print_("weight：", weight)

        # 反向传播0.75136507 0.77292847
        # 计算w5-w8（输出层权重）的误差
        pd_e_out_o1 = - (target1 - out_o1)  # 对o1输出的微积分
        pd_out_o1_net_o1 = sigmoid_derivation(out_o1)  # 对o1激活函数的积分
        pd_e_net_o1_net_o1 = pd_e_out_o1 * pd_out_o1_net_o1

        pd_net_o1_w5 = out_h1  # 对得到net_o1的函数中变量w5的微积分
        pd_e_w5 = pd_e_net_o1_net_o1 * pd_net_o1_w5  # 用整体的误差对w5求偏导
        pd_net_o1_w6 = out_h2
        pd_e_w6 = pd_e_net_o1_net_o1 * pd_net_o1_w6  # 用整体的误差对w6求偏导

        # 同理用同样的方法对w7和w8更新
        pd_e_out_o2 = - (target2 - out_o2)
        pd_out_o2_net_o2 = sigmoid_derivation(out_o2)
        pd_e_net_o2 = pd_e_out_o2 * pd_out_o2_net_o2

        pd_net_o2_w7 = out_h1
        pd_e_w7 = pd_e_net_o2 * pd_net_o2_w7
        pd_net_o2_w8 = out_h2
        pd_e_w8 = pd_e_net_o2 * pd_net_o2_w8

        # 计算w1-w4(输出层权重)的误差
        # 由h1输出out1到o1的net_o1 相当于 y=w*x + b
        # y为Net_o1 和x为Out_h1 然后对Out_h1微积分得到的值为w

        pd_net_o1_out_h1 = weight[5 - 1]  # 对Out_h1到Net_o1函数中输入值的
        pd_net_o2_out_h1 = weight[7 - 1]

        # 求出h1输出的误差占总的误差的比重
        pd_e_out_h1 = pd_e_net_o1_net_o1 * pd_net_o1_out_h1 + pd_e_net_o2 * pd_net_o2_out_h1  # (汇合了）
        pd_out_h1_net_h1 = sigmoid_derivation(out_h1)  # 对激活函数微积分
        pd_net_h1_w1 = i1
        pd_net_h1_w2 = i2

        # 参考公式
        pd_e_net_h1 = pd_e_out_h1 * pd_out_h1_net_h1
        pd_e_w1 = pd_e_net_h1 * pd_net_h1_w1
        pd_e_w2 = pd_e_net_h1 * pd_net_h1_w2
        pd_net_o1_out_h2 = weight[6 - 1]
        pd_net_o2_out_h2 = weight[8 - 1]
        pd_out_h2_net_h2 = sigmoid_derivation(out_h2)

        # 由输入的公式微积分y = w*x + b可知
        pd_net_h2_w3 = i1
        pd_net_h2_w4 = i2
        pd_e_out_h2 = pd_e_net_o1_net_o1 * pd_net_o1_out_h2 + pd_e_net_o2 * pd_net_o2_out_h2
        pd_e_net_h2 = pd_e_out_h2 * pd_out_h2_net_h2
        pd_e_w3 = pd_e_net_h2 * pd_net_h2_w3
        pd_e_w4 = pd_e_net_h2 * pd_net_h2_w4

        # 权重更新
        weight[1 - 1] = weight[1 - 1] - alpha * pd_e_w1
        weight[2 - 1] = weight[2 - 1] - alpha * pd_e_w2
        weight[3 - 1] = weight[3 - 1] - alpha * pd_e_w3
        weight[4 - 1] = weight[4 - 1] - alpha * pd_e_w4
        weight[5 - 1] = weight[5 - 1] - alpha * pd_e_w5
        weight[6 - 1] = weight[6 - 1] - alpha * pd_e_w6
        weight[7 - 1] = weight[7 - 1] - alpha * pd_e_w7
        weight[8 - 1] = weight[8 - 1] - alpha * pd_e_w8

    return


def create_by_super():
    """
    这是利用向量和矩阵强大的数学工具简化网络的构建过程，深刻体会到数学的奥妙
    :return:
    """
    w1_4 = [[0.15, 0.20], [0.25, 0.30]]  # 输出层的权重
    w5_8 = [[0.40, 0.45], [0.50, 0.55]]  # 权重矩阵的维度
    b1 = 0.35
    b2 = 0.60

    x = [0.05, 0.10]  # 初始化输入
    y = [0.01, 0.99]  # 初始化对应的输出label
    for n in range(num_iter):
        # 正向传播
        net_h = np.dot(w1_4, x) + b1
        out_h = sigmoid(net_h)  # 激活函数，第一层激励值

        net_o = np.dot(w5_8, out_h) + b2
        out_o = sigmoid(net_o)  # 第二层激励值
        # 计算损失，使用代价函数E = 1/(2)*sum[y-out_o]^2
        E = 0.5 * np.square(y - out_o).sum()
        # 输出迭代后的预测值
        print(str(n) + "：result: " + str(out_o[0]) + ",result: " + str(out_o[1]))
        # 输出误差，可以通过改变迭代次数来查看效果
        print(str(n) + "：error: " + str(E))

        if n == num_iter - 1:
            print("latest result ：" + str(out_o[0]) + ",result: " + str(out_o[1]))
            weight1_4 = np.reshape(np.transpose(w1_4), 4, -1)
            weight5_8 = np.reshape(np.transpose(w5_8), 4, -1)
            print_("weight：", (np.concatenate([weight1_4, weight5_8], axis=0).reshape(-1)).tolist())

        # multiply只是将对应位置相乘即可，就是简单的组团运算（简单的运算通过向量的线性（一对一的）表达）
        # 通过链式规则解决了最后一层的求导 dp_e_(包含两个值）对应的是
        # dp_e_out_o1*dp_out_o1_net_01(0.13849856)和dp_e_out_o2*dp_out_o2_net_o2(-0.03809824)
        pd_e_net_o = np.multiply(-(y - out_o), np.multiply(out_o, 1 - out_o))

        # delta1的关键一步是np.dot(np.array(w5_8).T, pd_e_net_o)，这个应该是矩阵求导公式所致，w2是一个矩阵，这是z2（向量）对z1的求导，中间隔着a1（向量）
        pd_e_net_h = np.multiply(np.dot(np.array(w5_8).T, pd_e_net_o), np.multiply(out_h, 1 - out_h))

        # 其中np.dot(np.array(w5_8).T, pd_e_net_o),就是dp_e_out_h，np.multiply(out_h, 1 - out_h)就是pd_out_h_net_h，这是行向量对行向量求导，
        # 所有根据链式规则，得到了pd_e_net_h；这不能不说很神奇，可背后就是求导（向量、矩阵之间的求导理论，这就是数学的魅力和作用）
        # 计算完权重的变化后（即delta），更新权重，delta也可以称为梯度的变化
        for i in range(len(w5_8)):
            # pd_e_net_o * out_h才是pd_e_w5_8，注意*的用法，其实是multiple
            w5_8[i] = w5_8[i] - alpha * np.multiply(pd_e_net_o[i], out_h)

        for i in range(len(w1_4)):
            # pd_e_net_h[i] * np.array(x)才是w1_4的梯度
            w1_4[i] = w1_4[i] - alpha * pd_e_net_h[i] * np.array(x)

    return


def create_by_torch():
    """
    利用pytorch构建，这个是目前计划中的终极方法，也是最激动人心的方法，可以看到torch的强大
    :return:
    """
    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    w1_4 = [[0.15, 0.20], [0.25, 0.30]]  # 输出层的权重
    w5_8 = [[0.40, 0.45], [0.50, 0.55]]  # 权重矩阵的维度
    b1 = 0.35
    b2 = 0.60

    x = [0.05, 0.10]  # 初始化输入
    y = [0.01, 0.99]  # 初始化对应的输出label

    # Create random Tensors to hold input and outputs.
    # Setting requires_grad=False indicates that we do not need to compute gradients
    # with respect to these Tensors during the backward pass.
    x = torch.tensor(x, device=device, dtype=dtype)
    y = torch.tensor(y, device=device, dtype=dtype)

    # Create weight Tensors for Numpy.
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Tensors during the backward pass.
    w1_4 = torch.tensor(w1_4, device=device, dtype=dtype, requires_grad=True)
    w5_8 = torch.tensor(w5_8, device=device, dtype=dtype, requires_grad=True)

    for n in range(num_iter):
        # Forward pass: compute predicted y using operations on Tensors; these
        # are exactly the same operations we used to compute the forward pass using
        # Tensors, but we do not need to keep references to intermediate values since
        # we are not implementing the backward pass by hand.
        # 正向传播
        net_h = x.dot(w1_4) + b1
        out_h = sigmoid(net_h)  # 激活函数，第一层激励值

        net_o = out_h.dot(w5_8) + b2
        out_o = sigmoid(net_o)  # 第二层激励值
        # 计算损失，使用代价函数E = 1/(2)*sum[y-out_o]^2
        E = 0.5 * torch.square(y - out_o).sum()
        # 输出迭代后的预测值
        print(str(n) + "：result: " + str(out_o[0]) + ",result: " + str(out_o[1]))
        # 输出误差，可以通过改变迭代次数来查看效果
        print(str(n) + "：error: " + str(E))

        if n == num_iter - 1:
            print("latest result ：" + str(out_o[0]) + ",result: " + str(out_o[1]))
            weight1_4 = torch.reshape(torch.transpose(w1_4), 4, -1)
            weight5_8 = torch.reshape(torch.transpose(w5_8), 4, -1)
            print_("weight：", (torch.concatenate([weight1_4, weight5_8], axis=0).reshape(-1)).tolist())
        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call w1.grad and w2.grad will be Tensors holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        E.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        # An alternative way is to operate on weight.data and weight.grad.data.
        # Recall that tensor.data gives a tensor that shares the storage with
        # tensor, but doesn't track history.
        # You can also use torch.optim.SGD to achieve this.
        # with torch.no_grad():
            # w1 -= learning_rate * w1.grad
            # w2 -= learning_rate * w2.grad
            #
            # # Manually zero the gradients after updating weights
            # w1.grad.zero_()
            # w2.grad.zero_()

    return


if __name__ == '__main__':
    create_by_classic()
    create_by_super()
    create_by_torch()
