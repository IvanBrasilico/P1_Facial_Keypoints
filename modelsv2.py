## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        # output size = (W-F)/S +1 = (224-5)/1 + 1 = 220
        # maxpool(2, 2) = 220/2 = 110
        self.conv2 = nn.Conv2d(32, 64, 3)
        # output size = (W-F)/S +1 = (110-3)/1 + 1 = 108
        # maxpool(2, 2) = 108/2 = 54
        self.conv3 = nn.Conv2d(64, 128, 3)
        # output size = (W-F)/S +1 = (54-3)/1 + 1 = 52
        # maxpool(2, 2) = 52/2 = 26
        self.conv4 = nn.Conv2d(128, 128, 3)
        # output size = (W-F)/S +1 = (26-3)/1 + 1 = 24
        # maxpool(2, 2) = 24/2 = 12
        self.conv5 = nn.Conv2d(128, 128, 1)
        # output size = (W-F)/S +1 = (12-1)/1 + 1 = 12
        # maxpool(2, 2) = 12/2 = 6
        # CONV OUTPUT = (128, 6, 6)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(128*6*6, 136*8)
        self.fc2 = nn.Linear(136*8, 136*2)
        self.output = nn.Linear(136*2, 136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool((F.relu(self.dropout2(self.conv1(x)))))
        x = self.pool((F.relu(self.dropout2(self.conv2(x)))))
        x = self.pool((F.relu(self.dropout2(self.conv3(x)))))
        x = self.pool((F.relu(self.dropout2(self.conv4(x)))))
        x = self.pool((F.relu(self.dropout3(self.conv5(x)))))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.dropout4(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = self.output(x)
        # x = self.avgpool(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
