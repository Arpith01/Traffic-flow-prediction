import sys
import tensorflow as tf
import numpy as np
from keras import layers
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input, Reshape, Add, Dense, Lambda ,Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras import backend
from keras.engine.topology import Layer
import glob

"""
An individual ResNet unit used to build the residual network
"""
def res_unit(X, n_filters, kernel_size):
    X_shortcut = X
    
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(n_filters, kernel_size, padding="same")(X)
    
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(n_filters, kernel_size, padding="same")(X)
    
    X = Add()([X,X_shortcut])
    
    return X

"""
Beginning node of a ResNet
"""
def res_start(X, n_filters, kernel_size):
    X = Conv2D(n_filters, kernel_size, padding = "same")(X)
    return X

"""
Ending node of a ResNet
"""
def res_end(X, n_filters, kernel_size):
    X = Conv2D(n_filters, kernel_size, padding = "same")(X)
    return X


"""
Individual ResNet component used to capture Trend, period and closeness
"""

def res_net(input_shape, n_res_units, n_filters, kernel_size):
    X_input = Input(input_shape)
    X = res_start(X_input, n_filters, kernel_size)
    
    for i in range(n_res_units):
        X = res_unit(X, n_filters, kernel_size)
    
    output = res_end(X, 2, kernel_size)
#     output = FusionLayer()([output])
#     output = Activation('tanh')(output)
    
    model = Model(X_input, output, name='res_net')
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics= ['accuracy'])
    return model

"""
Custom Later implementing weights for Fusion functionality
"""

class FusionLayer(Layer):
    def __init__(self, **kwargs):
        super(FusionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=input_shape[1:], initializer='uniform', trainable=True)
        super(FusionLayer, self).build(input_shape)
        
    def call(self, x):
        y = x*self.kernel
        return y
    
    def compute_output_shape(self, input_shape):
        return input_shape