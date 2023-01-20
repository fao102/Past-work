import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.colors import ListedColormap
from datetime import datetime

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

def plot_data(X, y, figsize=None):
    if not figsize:
        figsize = (8, 6)
    plt.figure(figsize=figsize)
    plt.plot(X[y==0, 0], X[y==0, 1], 'or', alpha=0.5, label=0)
    plt.plot(X[y==1, 0], X[y==1, 1], 'ob', alpha=0.5, label=1)
    plt.xlim((min(X[:, 0])-0.1, max(X[:, 0])+0.1))
    plt.ylim((min(X[:, 1])-0.1, max(X[:, 1])+0.1))
    plt.legend()




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

x_train_tensor = torch.tensor(X_train, dtype=torch.float, device="cpu")
y_train_tensor = torch.tensor(y_train, dtype=torch.float, device="cpu")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

plot_data(X_train, y_train)
