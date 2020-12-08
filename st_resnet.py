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
from modules import *

def st_resnet(input_shape, n_res_units, n_filters, kernel_size, mini, maxi):
    
    closeness_model = res_net((input_shape[0],input_shape[1], input_shape[2]//3), n_res_units, n_filters, kernel_size)
    period_model = res_net((input_shape[0],input_shape[1], input_shape[2]//3), n_res_units, n_filters, kernel_size)
    trend_model =  res_net((input_shape[0],input_shape[1], input_shape[2]//3), n_res_units, n_filters, kernel_size)
        
    fused_output = Add()([FusionLayer()([closeness_model.output]), FusionLayer()([period_model.output]), FusionLayer()([trend_model.output])])
    
    output = Activation('tanh')(fused_output)
    
#     output1 = output[:,:,:,:,0]
#     output2 = output[:,:,:,:,1]
    
#     o_shape = output.shape
#     print(o_shape)
    
# #     output1 = tf.reshape(output1,[1, o_shape[2]*o_shape[3]])
# #     output2 = tf.reshape(output2,[1, o_shape[2]*o_shape[3]])

# #     output = Lambda(lambda x: tf.reshape(x,[2, o_shape[2]*o_shape[3]]))(output)
    
#     output = Reshape((2, o_shape[2]*o_shape[3]))(output)
    
#     print(output.shape)
        
#     def scale(x):
#         return x*(scaler.data_range_) + scaler.data_min_

    
#     output = Lambda(scale)(output)
    
# #     output = Lambda(lambda x: tf.reshape(x,[o_shape[2],o_shape[3],2]))(output)
    
#     output = Reshape((o_shape[2],o_shape[3],2))(output)
    
#     print(output.shape)
    
    #Scaling tanH output to training data's distribution
    
    def scale(x):
        return x*(maxi-mini) + mini
    
    def scale_range(x):
        return ((x+1)*(maxi-mini)/2)+mini
    
#     print(maxi, mini)
    
    output = Lambda(scale_range)(output)
    
    st_resnet_model = Model([closeness_model.input, period_model.input, trend_model.input], output, name = "st_resnet")
    return st_resnet_model
