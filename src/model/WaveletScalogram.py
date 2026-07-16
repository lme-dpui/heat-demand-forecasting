# This script carries the CNN model structure and its training procedure
# import the necessary libraries

import torch
from torch import nn


class Wavelet_Scalogram_VGG(nn.Module):

    def __init__(self, num_freq, hours_lookback, num_channel, prediction_horizon, architecture):
        super(Wavelet_Scalogram_VGG, self).__init__()
        self.num_freq = num_freq  # this is the number of features
        self.hours_lookback = hours_lookback
        self.num_channel = num_channel
        self.output_size = prediction_horizon

        self.layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()

        h_w = num_freq

        # Convolution part
        convolution_data = architecture.convolution
        self.conv_layers.append(nn.Conv2d(self.num_channel,
                                          convolution_data.conv_layers[0],
                                          kernel_size=(convolution_data.kernel_size[0], convolution_data.kernel_size[0]),
                                          stride=convolution_data.stride_length[0], padding=convolution_data.padding))
        h_w = ((h_w - convolution_data.kernel_size[0] + 2 * 1) / convolution_data.stride_length[0]) + 1
        current_dim = convolution_data.conv_layers[0]

        if architecture.batch_norm:
            self.conv_layers.append(nn.BatchNorm2d(current_dim))

        if convolution_data.activation_function_conv == 'relu':
            self.conv_layers.append(nn.ReLU())
        elif convolution_data.activation_function_conv == 'leaky_relu':
            self.conv_layers.append(nn.LeakyReLU(negative_slope=0.3))

        if convolution_data.pooling:
            self.conv_layers.append(nn.MaxPool2d(kernel_size=convolution_data.max_pool_kernel[0], stride=convolution_data.max_pool_stride[0]))
            h_w = ((h_w - convolution_data.max_pool_kernel[0]) / convolution_data.max_pool_stride[0]) + 1


        for i in range(1, len(convolution_data.conv_layers)):
            self.conv_layers.append(nn.Conv2d(current_dim,
                                         convolution_data.conv_layers[i],
                                         kernel_size=(convolution_data.kernel_size[i], convolution_data.kernel_size[i]),
                                         stride=convolution_data.stride_length[i], padding=convolution_data.padding))
            h_w = ((h_w - convolution_data.kernel_size[i] + 2 * 1) / convolution_data.stride_length[i]) + 1
            current_dim = convolution_data.conv_layers[i]

            if architecture.batch_norm:
                self.conv_layers.append(nn.BatchNorm2d(current_dim))

            if convolution_data.activation_function_conv == 'relu':
                self.conv_layers.append(nn.ReLU())
            elif convolution_data.activation_function_conv == 'leaky_relu':
                self.conv_layers.append(nn.LeakyReLU(negative_slope=0.3))

            if convolution_data.pooling:
                self.conv_layers.append(nn.MaxPool2d(kernel_size=convolution_data.max_pool_kernel[i], stride=convolution_data.max_pool_stride[i]))
                h_w = ((h_w - convolution_data.max_pool_kernel[i]) / convolution_data.max_pool_stride[i]) + 1



        # Dense part
        dense_data = architecture.dense
        self.dense_layers.append(nn.Linear(int(current_dim * h_w * h_w),
                                     dense_data.dense_layers[0]))
        current_dim = dense_data.dense_layers[0]

        if dense_data.activation_function_dense == 'relu':
            self.dense_layers.append(nn.ReLU())
        elif dense_data.activation_function_dense == 'leaky_relu':
            self.dense_layers.append(nn.LeakyReLU(negative_slope=0.3))

        for i in range(1, len(dense_data.dense_layers)):
            self.dense_layers.append(nn.Linear(current_dim,
                                         dense_data.dense_layers[i]))
            current_dim = dense_data.dense_layers[i]
            if dense_data.activation_function_dense == 'relu':
                self.dense_layers.append(nn.ReLU())
            elif dense_data.activation_function_dense == 'leaky_relu':
                self.dense_layers.append(nn.LeakyReLU(negative_slope=0.3))

        self.dense_layers.append(nn.Dropout(dense_data.dropout))
        self.dense_layers.append(nn.Linear(current_dim, self.output_size))

    def forward(self, x):




        for layer in self.conv_layers:
            x = layer(x)

        x = x.reshape(x.size(0), -1)

        for layer in self.dense_layers[:-1]:
            x = layer(x)
        out = self.dense_layers[-1](x)

        return out
