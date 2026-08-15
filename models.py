## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Covolutional Layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 2)
        
        # Maxpooling Layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features = 36864, out_features = 1000) # The number of input gained by "print("Flatten size: ", x.shape)" in below
        self.fc2 = nn.Linear(in_features = 1000,    out_features = 1000)
        self.fc3 = nn.Linear(in_features = 1000,    out_features = 136) # the output 136 in order to having 2 for each of the 68 keypoint (x, y) pairs
        
        # Dropouts
        self.drop1 = nn.Dropout(p = 0.3)
        self.drop2 = nn.Dropout(p = 0.3)   
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # First - Convolution, Activation, Pooling + Dropout
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Second - Convolution, Activation, Pooling and Dropout
        x = self.pool(F.relu(self.conv2(x)))
        
        # Third - Convolution, Activation, Pooling and Dropout
        x = self.pool(F.relu(self.conv3(x)))
        
        # Forth - Convolution, Activation, Pooling and Dropout
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flattening the layer
        x = x.view(x.size(0), -1)
        
        # First - Dense, Activation and Dropout
        x = self.drop1(F.relu(self.fc1(x)))
        
        # Second - Dense, Activation and Dropout
        x = self.drop2(F.relu(self.fc2(x)))
        
        # Final Dense Layer
        x = self.fc3(x)
        
        return x
        
        
        
        
