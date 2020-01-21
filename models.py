## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

# custom Neural Network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:

        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting


        # First, define Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 512, 1)

        # Then we use maxpool layer with kernel_size = 2, stride = 2
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # We will apply Batch normalization after each Conv layer
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)


        # Series of fully-connected layer
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.fc2 = nn.Linear(1024, 136)

        # To avoid overfitting, we'll use Dropout after each FC with increasing probability
        self.fc1_drop = nn.Dropout(p=0.4)


        # Glorot uniform initialization (Xavier uniform initialization) for fully-connected layers
        # based on NaimishNet parepr: https://arxiv.org/pdf/1710.00977.pdf
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # forward behaviour for convolutional layers
        # blocks: CONV -> ReLU -> BN -> MAXPOOL
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = self.pool(x)

        # flatten layer
        x = x.view(x.size(0), -1)

        # forward behaviour for fully-connected layers
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)

        x = self.fc2(x)


        # a modified x, having gone through all the layers of your model, should be returned
        return x


# baseline network architecture
class BaseNet(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:

        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting


        # First, define Convolutional layers
        # first convolutional block
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 5)

        # second convolutional block
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)

        # third convolutional block
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)

        # fourth convolutional block
        self.conv7 = nn.Conv2d(128, 256, 1)
        self.conv8 = nn.Conv2d(256, 256, 1)

        # Then we use maxpool layer with kernel_size = 3, stride = 3
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)


        # Series of fully-connected layer
        self.fc1 = nn.Linear(256*11*11, 1024)
        self.fc2 = nn.Linear(1024, 136)

        # To avoid overfitting, we'll use Dropout after each FC with increasing probability
        self.conv1_drop = nn.Dropout(p=0.1)
        self.conv2_drop = nn.Dropout(p=0.1)

        self.conv3_drop = nn.Dropout(p=0.2)
        self.conv4_drop = nn.Dropout(p=0.2)

        self.conv5_drop = nn.Dropout(p=0.3)
        self.conv6_drop = nn.Dropout(p=0.3)

        self.conv7_drop = nn.Dropout(p=0.4)
        self.conv8_drop = nn.Dropout(p=0.4)

        self.fc1_drop = nn.Dropout(p=0.6)


        # Glorot uniform initialization (Xavier uniform initialization) for fully-connected layers
        # based on NaimishNet parepr: https://arxiv.org/pdf/1710.00977.pdf
        #nn.init.xavier_uniform_(self.fc1.weight)
        #nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # forward behaviour for convolutional layers
        # blocks: CONV -> BN -> CONV - MAXPOOL
        x = F.relu(self.conv1(x))
        x = self.conv1_drop(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.conv2_drop(x)

        x = F.relu(self.conv3(x))
        x = self.conv3_drop(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.conv4_drop(x)

        x = F.relu(self.conv5(x))
        x = self.conv5_drop(x)
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = self.conv6_drop(x)

        x = F.relu(self.conv7(x))
        x = self.conv7_drop(x)
        x = F.relu(self.conv8(x))
        x = self.pool(x)
        x = self.conv8_drop(x)

        # flatten layer
        x = x.view(x.size(0), -1)

        # forward behaviour for fully-connected layers
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)

        x = self.fc2(x)


        # a modified x, having gone through all the layers of your model, should be returned
        return x
