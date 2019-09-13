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
import sys

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
elif args.mode==1: transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
else:
    print("Unidentified mode.")
    sys.exit()
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
else: model = LeNet5()
##################################
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
training_accuracy = []

for epoch in range(num_epochs):
    print("Epoch:", epoch)
    for batch_num, train_batch in enumerate(train_loader):
        images, labels = train_batch
        ############################
        img = images.reshape(-1, 28 * 28) if args.mode == 0 else images.reshape(-1, 1, 28, 28)
        inputs = Variable(img)
        ############################
        targets = Variable(labels)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        accuracy = 0.0
        num_batches = 0
    for batch_num, training_batch in enumerate(train_loader):
        num_batches += 1
        images, labels = training_batch
        ##############################
        img = images.reshape(-1, 28 * 28) if args.mode == 0 else images.reshape(-1, 1, 28, 28)
        inputs = Variable(img)
        ##############################
        targets = labels.numpy()
        inputs = Variable(inputs)
        outputs = model(inputs)
        outputs = outputs.data.numpy()
        predictions = np.argmax(outputs, axis = 1)
        accuracy += accuracy_score(targets, predictions)
        final_acc = accuracy/num_batches
        training_accuracy.append(final_acc)

    print("Epoch: {} Training Accuracy: {}".format(epoch, final_acc*100))



## test on testing dataset
with torch.no_grad():
    correct = 0
    total = 0
    for test_batch in test_loader:
        images, labels = test_batch
        ###########################
        images = images.reshape(-1, 28 * 28) if args.mode == 0 else images.reshape(-1, 1, 28, 28)
        ###########################
        labels = labels
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))




##OPTIONAL display epochs and accuracy using Matplotlib
