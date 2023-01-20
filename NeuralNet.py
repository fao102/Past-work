import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from matplotlib.colors import ListedColormap
import numpy
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def plot_decision_boundary(func, X_tensor, y_tensor, figsize=(9, 6)):
    X = X_tensor.to("cpu").detach().numpy()
    y = y_tensor.to("cpu").detach().numpy()
    func.to("cpu")
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)
    
    aa, bb = np.meshgrid(hticks, vticks)
    ab = Variable(torch.from_numpy(np.c_[aa.ravel(), bb.ravel()]).float())
    c = func.pred(ab).detach().numpy()
    cc = c.reshape(aa.shape)

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    fig, ax = plt.subplots(figsize=figsize)
    contour = plt.contourf(aa, bb, cc, cmap=cm, alpha=0.8)
    
    ax_c = fig.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, 0.25, 0.5, 0.75, 1])
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.xlim(amin, amax)
    plt.ylim(bmin, bmax)

def plot_confusion_matrix(y, y_pred):
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(confusion_matrix(y, y_pred)), annot=True, fmt='d', cmap='YlGnBu', alpha=0.8, vmin=0)



def twospirals(n_points, noise=0):
    """
     Returns the two spirals dataset.
     https://glowingpython.blogspot.com/2017/04/solving-two-spirals-problem-with-keras.html
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))


def recommand_tanh(x):
  return 1.7159*torch.tanh((2/3)*x)


class FileNN(nn.Module):#inherits from nn.module
    def __init__(self) -> None:
        super().__init__() 
        self.layer_1_net = nn.Linear(2,800,bias=True) 
        self.layer_1_act = recommand_tanh
        self.layer_1_1_act = recommand_tanh
        self.layer_1_1_net = nn.Linear(800,800,bias=True)
        self.layer_2_act = nn.Sigmoid()
        self.layer_2_net = nn.Linear(800,1,bias=True)#f(linear)=Wx +b

    def train(self, data, num_epochs = 10, lr = 0.003):
        criterion = nn.BCELoss()#loss function
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)#using gradient descent, >0 = left and <0 = right
        for epoch in range(num_epochs):
            for item in data:
                out = self(item[0])
                optimizer.zero_grad()
                loss = criterion(torch.squeeze(out), item[1])
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


X, y = twospirals(500)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

learning_rate = 0.1
batch_size = 32
num_epochs = 850



x_train_tensor = torch.tensor(X_train, dtype=torch.float, device="cpu")
y_train_tensor = torch.tensor(y_train, dtype=torch.float, device="cpu")
dataset = TensorDataset(x_train_tensor, y_train_tensor)
loader = DataLoader(dataset, batch_size = batch_size)


FileNet = FileNN()
FileNet.to("cpu")
strat_time = datetime.now()
FileNet.train(loader,num_epochs=num_epochs, lr=learning_rate)
end_time = datetime.now()

print('time spend in the training: ', end_time - strat_time)
plot_decision_boundary(FileNet, x_train_tensor, y_train_tensor)


x_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

test_pred = FileNet.pred(x_test_tensor)
test_pred_np = test_pred.detach().numpy()

num_correct = np.sum(y_test == test_pred_np)
print("accuracy: ", num_correct/len(y_test))

classification_results = classification_report(y_test, test_pred_np)
print(classification_results)
plot_confusion_matrix(y_test, test_pred_np)