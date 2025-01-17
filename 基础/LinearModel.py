import torch
import numpy as np
import matplotlib.pyplot as plt

#数据集
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

#前馈函数:乘以权重w
def forward(x):
    return x*w

#损失函数:计算损失
def loss(x,y):
    y_pred = forward(x)
    return(y_pred-y)*(y_pred-y)
#穷举法:把权重从0.0到4.1依次尝试,间隔0.1   w_list是空数组用来存放权重值   mse_list用来放对用权重下的平均平方误差(MSE)即平均损失
w_list=[]
mse_list=[]
for w in np.arange(0.0,4.1,0.1):
    print("w=",w)
    l_sum=0
    #evaluate(估值)即x,y带入前馈函数后得到的数据
    for x_val,y_val in zip(x_data,y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val,y_val)
        l_sum += loss_val
        print('\t',x_val,y_val,y_pred_val,loss_val)
    print('MSE=',l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)
