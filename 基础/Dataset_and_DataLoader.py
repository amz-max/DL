

# for epoch in range(100):      一次epoch是所有样本的一次前馈反馈和更新的过程
#     for i in range(total_batch)       每次迭代的是batch
                                    #batch_size是每次进行前馈反馈更新的样本数量即每个batch内的样本数量
                                    #iteration:batch的数量

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset #(抽象类,不能实例化)
from torch.utils.data import DataLoader


#数据准备通式
# class MyDataset(Dataset):
#     def __init__(self):
#         pass
#     def __getitem__(self, index):   #有该方法后可通过索引取出数据
#         pass
#     def __len__(self):      #可以返回数据里面的数据数量
#         pass
#
# dataset = MyDataset()
# train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)
#                         #数据位置, 每块batch的大小,是否每次分batch时把数据打乱,并行的进程

#if__name__=='__main__':    报错时在循环前加这句话



#数据准备
class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath, delimiter=',',dtype=np.float32)
                        #文件位置,用逗号作为分隔符,读取32位浮点数
        self.len=xy.shape[0]    #拿到数据的行数,即数据集的个数
        self.x_data=torch.from_numpy(xy[:,:-1])     #要前八列
        self.y_data=torch.from_numpy(xy[:,[-1]])    #只要最后一列

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.len

dataset = DiabetesDataset('diabetes.csv.gz')    #数据文件路径
train_loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)   #!--面的都是为了这一句准备--!


#模型设计
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

model = Model()     #!--设计了有关多维特征输入且多层的前馈--!

#优化器和损失器
criterion = torch.nn.BCELoss(size_averages=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


#训练循环
for epoch in range(100):
    for i,data in enumerate(train_loader,0):
        #Prepare data
        inputs,labels = data    #将x,y从数据中拿出来,并且根据batch_size的大小,将拿出来的足够的x,y自动转换为张量
        #Forward
        y_pred = model(inputs)
        loss = criterion(y_pred,labels)
        print(epoch,i,loss.item())
        #Backward
        optimizer.zero_grad()
        loss.backward()
        #Update
        optimizer.step()

