import torch.nn as nn


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)


        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                    )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2))
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        ##CONV BLOCK 1
        x = self.conv1(x)
        ##CONV BLOCK 2
        x = self.conv2(x)
        ##FULLY CONNECTED BLOCK
        x = x.view(-1, 16*5*5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
