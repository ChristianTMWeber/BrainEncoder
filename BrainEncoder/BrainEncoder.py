# pytorch
import torch
from torch import nn

# torchsummary for easy checking and debugging
from torchinfo import summary

import numpy as np

# Neural Network Model parameters
# maybe later make them dynamically changeable
nLayers = 3
kernelSizes = [3, 3, 3]
nChannels = [32, 64, 128]
kernelStrides = [2, 2, 2]
latendSpaceDim = 8192

inputChannelsize = 1 # for now we presume only 1 color channel

class Encoder(nn.Module):
    
    def __init__(self, output_dim=2, use_batchnorm=False):
        super(Encoder, self).__init__()
        
        # bottleneck dimentionality
        self.output_dim = output_dim

        self.use_batchnorm = use_batchnorm
    
        # convolutional layer hyper parameters
        self.layers = nLayers
        self.kernels = kernelSizes
        self.channels = nChannels
        self.strides = kernelStrides
        self.conv = self.define_Convolutional_Layers()
        
        # layers for latent space projection
        self.fullyConnectedDim = latendSpaceDim
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.fullyConnectedDim, self.output_dim)
    
    
    def define_Convolutional_Layers(self):
        """
        Setup the convolutional layers for the NN
        """
        conv_layers = nn.Sequential()
        for i in range(self.layers):
            
            # for the first layer we match the input channel size to the input, e.g. #ColorChannels in input, e.g. 1 for now
            if i == 0: conv_layers.append( nn.Conv3d( inputChannelsize ,   self.channels[i], 
                                              kernel_size=self.kernels[i], stride=self.strides[i],
                                              padding=1))

            # for the other layers the input size is given by the output size of the prior layer            
            else: conv_layers.append(nn.Conv3d(self.channels[i-1],    self.channels[i],
                                         kernel_size=self.kernels[i], stride=self.strides[i],
                                         padding=1))

            # we can normalize the mean and standard deviation of the conv. layer to improve learning stability
            if self.use_batchnorm: conv_layers.append(nn.BatchNorm3d(self.channels[i]))
            
            # GELU - Gaussian Error Linear Units for activation function
            conv_layers.append(nn.GELU()) 

        return conv_layers
    
    
    def forward(self, x_in):
        x_conv = self.conv(x_in)
        x_flat = self.flatten(x_conv)
        return self.linear(x_flat)


class Decoder(nn.Module):
    
    def __init__(self, input_dim=2, use_batchnorm=False):
        super(Decoder, self).__init__()
        
        self.use_batchnorm = use_batchnorm

        self.fullyConnectedDim = latendSpaceDim
        self.input_dim = input_dim
        
        # Conv layer hypyer parameters
        self.layers = nLayers
        self.kernels = kernelSizes
        self.channels = nChannels[::-1] # flip the channel dimensions
        self.strides = kernelStrides
        
        # In decoder, we first do fc project, then conv layers
        self.linear = nn.Linear(self.input_dim, self.fullyConnectedDim)
        self.conv =  self.define_Convolutional_Layers()

        self.output = nn.Conv3d(self.channels[-1], 1, kernel_size=1, stride=1)

    def define_Convolutional_Layers(self):
        conv_layers = nn.Sequential()
        for i in range(self.layers):
            
            if i == 0: conv_layers.append(
                            nn.ConvTranspose3d(self.channels[i], self.channels[i],
                                               kernel_size=self.kernels[i], stride=self.strides[i],
                                               padding=1,output_padding=1)
                            )
            
            else: conv_layers.append(
                            nn.ConvTranspose3d(self.channels[i-1], self.channels[i],
                                               kernel_size=self.kernels[i], stride=self.strides[i],
                                               padding=1, output_padding=1
                                              )
                            )
            
            if self.use_batchnorm and i != self.layers - 1:
                conv_layers.append(nn.BatchNorm3d(self.channels[i]))

            conv_layers.append(nn.GELU())

        return conv_layers
    
    
    def forward(self, x_in):
        x_lin = self.linear(x_in)
        # reshape 4D tensor to 5D tensor
        x_reshaped = x_lin.reshape(x_lin.shape[0], 128, 4, 4, 4)
        x_conv = self.conv(x_reshaped)
        return self.output(x_conv)

class AutoEncoder(nn.Module):
    
    def __init__(self, latentSpaceSize = 2):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(output_dim=latentSpaceSize, 
                               use_batchnorm=True)
        self.decoder = Decoder(input_dim=latentSpaceSize,
                               use_batchnorm=True)
        
    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == "__main__":


    testAutoEncoder = AutoEncoder()

    print(summary(testAutoEncoder))


    
    print("Loaded Neural Network correctly")