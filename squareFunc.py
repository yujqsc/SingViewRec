import torch
import torch.nn.functional as f
from torch.autograd import Variable
import matplotlib.pyplot as plt

"""生成随机数据"""
x = torch.unsqueeze(torch.linspace(0, 20, 20), dim=1)  # x轴数据从-1到1，共100个数据，unsqueeze把一维的数据变为2维的数据
y = x.pow(2) + 0.2 * torch.rand(x.size())  # y=x*2，但是还要加上波动
x, y = Variable(x), Variable(y)  # 变成Variable的形式，神经网络只能输入Variable

"""创建神经网络"""


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):  # 初始化信息
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 一层隐藏层神经网络，输入信息，输出信息（神经元个数）
        self.hidden1 = torch.nn.Linear(n_hidden, 2*n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 一层预测层神经网络，输入信息，输出信息（神经元个数）
    def forward(self, x):  # 前向传递，x为输入信息
        x = f.leaky_relu(self.hidden(x),0.2)  # 出了隐藏层要激活
        #x = f.leaky_relu(self.hidden1(x), 0.2)
        x = self.predict(x)
        return x

class Filter(torch.nn.Module):
    def __init__(self):
        super(Filter,self).__init__()
        kernel=[1,-2,1]
        kernel=torch.FloatTensor(kernel).cuda()
        kernel=kernel.expand(1,-1)
        self.weight=torch.nn.Parameter(data=kernel,requires_grad=False)
    def forward(self,x):
        return torch.nn.functional.conv1d(x,weight=self.weight,padding=1)
"""打印处理数据"""
net = Net(1, 10, 1)  # 输入值1个x，隐藏层10个，输出值一个y
plt.ion()  # 实时打印的过程
plt.show()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.0006)  # 使用优化器优化神经网络参数，lr为学习效率，SGD为随机梯度下降法
loss_func = torch.nn.L1Loss()  # 均方差处理回归问题
for t in range(30000):
    if t==2400:# 循环训练
        for p in optimizer.param_groups:
            p['lr'] *= 0.01
    prediction = net(x)  # 输入x，得到预测值
    loss = loss_func(prediction, y)  # 计算损失，预测值和真实值的对比
    optimizer.zero_grad()  # 梯度先全部降为0
    loss.backward()  # 反向传递过程
    # for name, parms in net.named_parameters():
    #     if t % 1 == 0:
    #         if name == 'hidden.weight':
    #             print('层:', name, parms.size(), '-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #                   ' -->grad_value:', parms.grad)
    optimizer.step()  # 以学习效率0.5来优化梯度
    """循环打印"""
    if t % 5 == 0:  # 每五步打印一次
        plt.cla()  # 清除当前图形中的当前活动轴。其他轴不受影响
        plt.scatter(x.data.numpy(), y.data.numpy())  # 打印原始数据
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)  # 打印预测数据
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})  # 打印误差值
        plt.pause(0.1)  # 每次停顿0.1
plt.ioff()
plt.show()