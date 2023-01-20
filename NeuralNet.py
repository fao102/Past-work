import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from matplotlib.colors import ListedColormap
import numpy
from datetime import datetime

def recommand_tanh(x):
  return 1.7159*torch.tanh((2/3)*x)


class FileNN(nn.Module):#inherits from nn.module
    def __init__(self) -> None:
        super().__init__(self) 
        self.layer_1_net = nn.Linear(2,800,bias=True) 
        self.layer_1_act = recommand_tanh
        self.layer_1_1_activation = recommand_tanh
        self.layer_1_1_net = nn.Linear(800,800,bias=True)
        self.layer_2_act = nn.Sigmoid()
        self.layer_2_net = nn.Linear(800,800,bias=True)#f(linear)=Wx +b

    def train(self, data, num_epochs = 10, lr = 0.003):
        criterion = nn.BCELoss()#loss function
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)#using gradient descent, >0 = left and <0 = right
        for epoch in range(num_epochs):
            for item in data:
                out = self(item[0])
                optimizer.zero_grad()
                loss = criterion(torch.squeeze(out, item[1]))
                loss.backward()
                optimizer.step()

    
    def forward(self,x):
        # x is our input, we pass the input to the layer_1_net linear transformation
        net_1_linear_trans = self.layer_1_net(x)
        #After we obtain the linear transformation, we pass the net_1_linear_transformation to the activation function.
        net_1_act = self.layer_1_act(net_1_linear_trans)

        net_1_1_linear_trans = self.layer_1_1_net(net_1_act)
        net_1_1_act = self.layer_1_1_act(net_1_1_linear_trans)
    
        # similar to layer 1 linear transformation, now we conduct linear transformation from hidden layer to output layer
        net_2_linear_trans = self.layer_2_net(net_1_1_act)
        # Apply Sigmoid activation function
        layer2Act = self.layer_2_act(net_2_linear_trans)
        return layer2Act
    
    

    def pred(self, data, thres = 0.5):
        pred = self.forward(data)
        ans = []
        for t in pred:
            if t < thres:
                ans.append(0)
            else:
                ans.append(1)

        return torch.tensor(ans)



learning_rate = 0.003
batch_size = 32
num_epochs = 500

dataset = TensorDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size = batch_size)

FileNet = FileNN()
FileNet.to("cpu")
strat_time = datetime.now()
FileNet.train(loader,num_epochs=num_epochs, lr=learning_rate)
end_time = datetime.now()
