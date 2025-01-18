import torch
import torchvision

x_data = [[1.0],[2.0],[3.0]]
y_data = [[0],[0],[1]]

import torch.nn.functional as F

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))      #logist函数是sigmod中最出名的,把数值缩放到0-1,表示概率
        return y_pred
model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)  #求y_pred
    loss = criterion(y_pred,y_data) #求loss
    print(epoch,loss.item())

    optimizer.zero_grad()   #清零梯度
    loss.backward()     #反向传播
    optimizer.step()    #更新参数