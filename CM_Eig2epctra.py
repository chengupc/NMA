'''
Chengch
2017-12-13

'''
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from time import time
from sys import argv 
import matplotlib
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
#
#script, png = argv

# Hyper Parameters 
hidden_size1 = 1000
hidden_size2 = 1000
hidden_size3 = 1000
num_epochs = 5 
batch_size1 = 1000 
learning_rate = 0.001

#data loader
Eig = np.load('/home/chengch/12-14/Eig_CM_08.npy')[:2000]
num = len(Eig)
split_point = int(num / 10 * 9) # the split_point for test_data and train_data

Eig = np.load('/home/chengch/12-14/Eig_CM_08.npy')[:2000]
IR = np.load('/home/chengch/12-14/Fre_08.npy')[:2000]

total_data = []
for x,y in zip(Eig,IR):
    total_data.append((x,y))
np.random.shuffle(total_data)

train_data = total_data[:split_point]
test_data = total_data[split_point:]
batch_size2 = len(test_data)
training_data =[]
testing_data = []

for x1,y1 in train_data:
    training_data.append((torch.FloatTensor(x1),y1)) 
for x2,y2 in test_data:
    testing_data.append((torch.FloatTensor(x2),y2))
# Data Loader (Input Pipeline)

train_loader = torch.utils.data.DataLoader(dataset=training_data, 
                                           batch_size=batch_size1, 
                                           shuffle=True)
#
#test_loader = torch.utils.data.DataLoader(dataset=testing_data, 
#                                          batch_size=batch_size2, 
#                                          shuffle=True)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2,hidden_size3, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)  
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, num_classes)  
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out
net = Net(len(x1), hidden_size1, hidden_size2,hidden_size3, 1 )

# Loss and Optimizer
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
# Train the Model
RMSD =[]
loss_set = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): 
        images = Variable(images)
        labels = Variable(torch.FloatTensor(labels.numpy()))
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss1 = criterion(outputs, labels) 
        loss1.backward()
        optimizer.step()
    loss_set.append((loss1.data[0])** 0.5)
    test_input,test_label = np.array(test_data)[:,0],np.array(test_data)[:,1]
    test_input = Variable(torch.FloatTensor(test_input))
    test_label = Variable(torch.FloatTensor(test_label))
    prediction = net(test_input)
    loss2 = criterion(prediction,test_label)
    RMSD.append((loss2.data[0])** 0.5)
    print (epoch)
    
#figure(figsize=(9,6))
#a = np.arange(len(RMSD))[10:]
#plt.plot(a,loss_set[10:],'r-')
#plt.plot(a,RMSD[10:],'b-')
#savefig(png,dpi=200)
#plt.show()
