# %%
import h5py
import time as t
import numpy as np
import scipy as sp
import scipy.io as spi
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
torch.manual_seed(0)

tsamp = 60000

hln = 300
name = 'D3W300'
act = nn.ReLU()
act_name = 'ReLU'

data = spi.loadmat('data/AD_5000_DP_TrData.mat')

u_train = data['u_in'][0:tsamp,:]
x_t_train = data['x_t_in'][0:tsamp,:]
s_train = data['s_in'][0:tsamp,:]

u_test = data['u_in'][tsamp:-1,:]
x_t_test = data['x_t_in'][tsamp:-1,:]
s_test = data['s_in'][tsamp:-1,:]

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class model1(nn.Module):
    def __init__(self,hln,act):
        super(model1, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, hln),
            act,
            nn.Linear(hln, hln),
            act,
            nn.Linear(hln, hln),
            act)
                    
    def forward(self, x):
        y =  self.layers(x)
        return y

class model2(nn.Module):
    def __init__(self,hln,act):
        super(model2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, hln),
            act,
            nn.Linear(hln, hln),
            act,
            nn.Linear(hln, hln),
            act)
                    
    def forward(self, x):
        y =  self.layers(x)
        return y


class lastlayer(nn.Module):
    def __init__(self,):
        super(lastlayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1, 1))
                    
    def forward(self, x):
        y =  self.layer(x)
        return y


branch_model = model1(hln,act)
trunk_model = model2(hln,act)
lastmodel = lastlayer()
models = [branch_model,trunk_model,lastmodel]

def train(models,xtrain,ytrain,xtest,ytest):
    optimizer = torch.optim.Adam(list(models[0].parameters())+list(models[1].parameters())+list(models[2].parameters()))

    def fn(x):
        y = torch.einsum("ij, ij->i", x[0], x[1])
        y = torch.unsqueeze(y, axis = 1)
        return y
    criteria = nn.MSELoss()
    epochs = 10000
    loss_store = np.zeros(epochs)
    test_loss_store = np.zeros(epochs)
    for epoch in range(epochs):
        # prediction
        y1 = models[0](torch.from_numpy(xtrain[0]).float())
        y2 = models[1](torch.from_numpy(xtrain[1]).float())
        combined = fn([y1,y2])
        y = models[2](combined)

        # loss
        loss = criteria(y,torch.from_numpy(ytrain).float())
        loss_store[epoch] = loss.detach().numpy()

        # test
        y1 = models[0](torch.from_numpy(xtest[0]).float())
        y2 = models[1](torch.from_numpy(xtest[1]).float())
        combined = fn([y1,y2])
        y = models[2](combined)

        # loss
        test_loss = criteria(y,torch.from_numpy(ytest).float())
        test_loss_store[epoch] = test_loss.detach().numpy()

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save model
        if epoch == 0:
            minloss = loss
        elif loss < minloss:
            minloss = loss
            torch.save(models[0].state_dict(), 'MLP_model/model'+name+act_name+'_info1')
            torch.save(models[1].state_dict(), 'MLP_model/model'+name+act_name+'_info2')
            torch.save(models[2].state_dict(), 'MLP_model/model'+name+act_name+'_info3')
            print('model saved')
        print('loss:{}, epoch:{}'.format(loss,epoch))
    np.save('result/MLP_loss'+name+act_name,loss_store)
    np.save('result/MLP_test_loss'+name+act_name,test_loss_store)
    
# Train
train(models,[u_train,x_t_train],s_train,[u_test,x_t_test],s_test)

# Prediction




