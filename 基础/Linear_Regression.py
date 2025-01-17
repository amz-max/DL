import torch

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):     #Module是一个类,可以自动求反向传播;而Functions没有,需要设计如何求反向传播
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)  #这是一个类,构造了一个参数为(1,1)的对象,包含权重和偏置
                                                                #可以自动完成wx+b的过程,且Linear也继承自Module,可以完成自动求导

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()   #实例化,创建一个model

criterion = torch.nn.MSELoss(size_average=False)    #损失函数
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)     #优化器:把权重参数更新
                                                            #model.parameters()自动找需要的权重做优化,lr是学习率



# 训练过程
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())


    optimizer.zero_grad()   #所有权重都归零
    loss.backward()     #反向传播
    optimizer.step()    #更新参数

#打印权重
print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())
#测试
x_test = torch.Tensor([4.0])
y_test = model(x_test)
print('y_pred=',y_test.data)