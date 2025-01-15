import torch

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w=torch.Tensor([1.0])
w.requires_grad=True   #表明需要计算梯度,因为默认Tensor不会去计算梯度

def forward(x):
    return w*x   #x不一定是Tensor,这里将x自动类型转换为Tensor
                 #由于w需要计算梯度,所以w*x得到的结果y_pred也需要计算梯度

def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2    #此处同样需要计算梯度

print('predict (before training)',4,forward(4).item())

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l=loss(x,y)     #前馈过程,计算出来后是Tensor类型
        l.backward()    #调用张量的成员函数,可以把所有需要梯度的地方都求出来   每一次反向传播(backward)后会把计算图释放,把得到的梯度值存到w中
        print('\tgrad:',x,y,w.grad.item())
        w.data=w.data-0.05*w.grad.data  #更新权重这里是纯数值计算,.data后就不会建立运算图,
                                        # 取w的梯度要用w.grad.data因为w的梯度也是一个张量所以一定要.data

        w.grad.data.zero_()     #把权重中储存的梯度的数据全部清零,否则下次计算时储存的梯度会和新的梯度数据运算,且必须手动清零

    print('progress:',epoch,l.item())   #.item()可以把张量转化为标量

print('predict(after training)',4,forward(4).item())



