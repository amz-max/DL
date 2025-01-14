import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w=1.0

def forward(x):
    return w*x

#算出整个训练数据的平均损失,在得到导数之和的结果中用不到该函数,只是为了在输出时可以直观地看到损失值在不断减小
def cost(xs,ys):
    cost = 0
    for x,y in zip(xs,ys):
        y_pred = forward(x)
        cost += (y_pred - y)**2
    return cost / len(xs)

#每个数据的损失函数对w求导,并对导数求和,返回平均导数
def gradient(xs,ys):
    grad = 0
    for x,y in zip(xs,ys):
        grad += 2 * x * (x * w - y)  #求导后的函数
    return grad / len(xs)

# 用数据4做例子,查看训练前后的差距
print('Predict (before training)',4,forward(4))

#进行100轮训练
for epoch in range(100):
    cost_val = cost(x_data,y_data)
    grad_val = gradient(x_data,y_data)
    w -= 0.05*grad_val  #0.05是学习速率,grad_val如果是负数:则w将加 学习速率*grad_val,从而向正确方向更新w,进一步降低损失值
                        #如果grad_val是正数:w将减去它,从而朝反方向更新w值,降低损失值
    print('Epoch:',epoch,'w=',w,'loss=',cost_val)
print('Predict (after training)',4,forward(4))