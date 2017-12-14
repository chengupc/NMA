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

batch_size1 = 100
num_epochs = 1000 
learning_rate = 0.0001

script, png = argv

CM = np.load('../data/CMs_sort_08.npy')
num = len(CM)
split_point = int(num / 10 * 9) # the split_point for test_data and train_data

CM = np.load('../data/CMs_sort_08.npy')
IR = np.load('../data/Fre_08.npy')

total_data = []
for x,y in zip(CM,IR):
    total_data.append((x,y))
np.random.shuffle(total_data)

train_data = total_data[:split_point]
test_data = total_data[split_point:]
batch_size2 = len(test_data)
training_data =[]
testing_data = []

for x1,y1 in train_data:
    x1 = np.reshape(x1,(1,27,27))
    training_data.append((torch.FloatTensor(x1),y1)) 
for x2,y2 in test_data:
    x2 = np.reshape(x2,(1,27,27))
    testing_data.append((torch.FloatTensor(x2),y2))
# Data Loader (Input Pipeline)

train_loader = torch.utils.data.DataLoader(dataset=training_data, 
                                           batch_size=batch_size1, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testing_data, 
                                          batch_size=batch_size2, 
                                          shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 27, 27)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 27, 27)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=3),    # choose max value in 3x3 area, output shape (16, 9, 9)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 9, 9)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 9, 9)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(3),                # output shape (32, 3, 3)
        )
        self.out = nn.Linear(32 * 3 * 3, 1)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 3 * 3)
        output = self.out(x)
        return output   # return x for visualization

cnn = CNN()
cnn = cnn.cuda()

criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)  

RMSD =[]
loss_set = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): 
        images = Variable(images).cuda()
        labels = Variable(torch.FloatTensor(labels.numpy())).cuda()
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = cnn(images)
        loss1 = criterion(outputs, labels) 
        loss1.backward()
        optimizer.step()
    loss_set.append((loss1.data[0])** 0.5)
    for _,(images,labels) in enumerate(test_loader):
        images = Variable(images).cuda()
        labels = Variable(torch.FloatTensor(labels.numpy())).cuda()
        prediction = cnn(images)
        loss2 = criterion(prediction,labels) 
    RMSD.append((loss2.data[0])** 0.5)
    print (epoch)
figure(figsize=(9,6))
a = np.arange(len(RMSD))[100:]
plt.plot(a,loss_set[100:],'r-')
plt.plot(a,RMSD[100:],'b-')
savefig(png,dpi=200)
plt.show()
