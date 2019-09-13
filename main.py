#BUILT IN IMPORTS
import argparse

#THIRD PARTY IMPORTS
import torch
import numpy as np
import torch.nn as nn
import torch.optim
import torchvision.datasets
import torchvision.transforms
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

#USER DEFINED IMPORTS
from models.lenet5 import LeNet5
from models.fullyconnected import FCNet

parser = argparse.ArgumentParser(description = 'enter algorithm/mode of choice')
parser.add_argument('--mode', type=int, help="Enter mode = 0 for Fully connected and mode = 1 for CNN")
args = parser.parse_args()

batch_size = 32
num_epochs = 2
#################################
###CONFLICT ZONE 1
if args.mode==0: transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor() ])
if args.mode==1: transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#################################
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transformImg,
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transformImg)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
##################################
###CONFLICT ZONE 2
if args.mode==0: model = FCNet()
if args.mode==1: model = LeNet5()
##################################
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
training_accuracy = []
for epoch in range(num_epochs):
    print("Epoch:", epoch)
    for batch_num, train_batch in enumerate(train_loader):
        images, labels = train_batch
        ############################
        inputs = Variable(images.reshape(-1, 1,28,28))
        ############################
        targets = Variable(labels)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
