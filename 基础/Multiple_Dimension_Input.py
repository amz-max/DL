import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model.self).__init__()
        self.linear1 = torch.nn.Linear(8,6) #维度由[n,8]变成[n,6],通过线性回归自动构造的w完成
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.activate = torch.nn.Sigmoid()


        def forward(self,x):
            x=self.activate(self.linear1(x))
            x=self.activate(self.linear2(x))
            x=self.activate(self.linear3(x))
            return x

model = Model()