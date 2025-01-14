import torch

#随机梯度下降算法是随机抽取一组或几组数据算梯度而不是将所有数据都算出来(该案例由于数据少所以每组数据都进行了计算),这样可以避免梯度消失的情况
# 但是不能并行导致其时间复杂度过高,但是性能比一般梯度下降算法好

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
w=1.0
def forward(x):
    return w*x

def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2

def gradient(x,y):
    return 2*x*(x*w-y)

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        grad=gradient(x,y)
        w=w-0.05*grad
        print('\tgrad:',x,y,grad)
        l=loss(x,y)

    print('progress:',epoch,'w=',w,'loss=',l)
print('after training',4,forward(4))