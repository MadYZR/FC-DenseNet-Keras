# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 13:08:47 2018

@author: Ryan_ye

refer to https://github.com/SimJeg/FC-DenseNet/blob/master/FC-DenseNet.py
"""

from keras.layers import Activation,MaxPooling2D,UpSampling2D,Dense,BatchNormalization,Input,Reshape,multiply,add,Dropout,AveragePooling2D,GlobalAveragePooling2D,concatenate
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.models import Model														  
import keras.backend as K
from keras.regularizers import l2
from keras.engine import Layer,InputSpec
from keras.utils import conv_utils

from layers import BN_ReLU_Conv, TransitionDown, TransitionUp, SoftmaxLayer

def Tiramisu(
        input_shape=(None,None,3),
        n_classes = 1,
        n_filters_first_conv = 48,
        n_pool = 5,
        growth_rate = 16 ,
        n_layers_per_block = [4,5,7,10,12,15,12,10,7,5,4],
        dropout_p = 0.2
        ):
    if type(n_layers_per_block) == list:
            print(len(n_layers_per_block))
    elif type(n_layers_per_block) == int:
            n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError
        
#####################
# First Convolution #
#####################        
    inputs = Input(shape=input_shape)
    stack = Conv2D(filters=n_filters_first_conv, kernel_size=3, padding='same', kernel_initializer='he_uniform')(inputs)
    n_filters = n_filters_first_conv

#####################
# Downsampling path #
#####################     
    skip_connection_list = []
    
    for i in range(n_pool):
        for j in range(n_layers_per_block[i]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            stack = concatenate([stack, l])
            n_filters += growth_rate
        
        skip_connection_list.append(stack)        
        stack = TransitionDown(stack, n_filters, dropout_p)
    skip_connection_list = skip_connection_list[::-1]

    
#####################
#    Bottleneck     #
#####################     
    block_to_upsample=[]
    
    for j in range(n_layers_per_block[n_pool]):
        l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
        block_to_upsample.append(l)
        stack = concatenate([stack,l])
    block_to_upsample = concatenate(block_to_upsample)

   
#####################
#  Upsampling path  #
#####################
    for i in range(n_pool):
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i ]
        stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)
        
        block_to_upsample = []
        for j in range(n_layers_per_block[ n_pool + i + 1 ]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = concatenate([stack, l])
        block_to_upsample = concatenate(block_to_upsample)

#####################
#  Softmax          #
#####################
    output = SoftmaxLayer(stack, n_classes)            
    model=Model(inputs = inputs, outputs = output)    
    model.summary()
    
    return model
    
    
