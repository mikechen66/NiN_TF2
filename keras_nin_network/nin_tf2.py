#!/usr/bin/env python
# coding: utf-8

"""
The script realizes the layers with the complete recursive function style with the new
tensorflow.keras. 

Please pay more attention on the formal argument "x". To faciliate the process of argument
passing during the function call in the context, we select x to express the recursion that 
is typical in the mathematics. 

Input() initizate a 3-D shape(weight,height,channels) into a 4-D tensor(batch,weight,height, 
channels). If no batch size, it is defaulted as None.

Reference:
https://arxiv.org/pdf/1312.4400.pdf
"""

from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def nin(input_shape):
    # Define Network in Network model
    input = Input(shape=input_shape)
    x = Conv2D(filters=192, kernel_size=(5,5), activation='relu')(input)
    x = Conv2D(filters=160, kernel_size=(1,1), activation='relu')(x)
    x = Conv2D(filters=96, kernel_size=(1,1), activation='relu')(x)
    x = MaxPool2D(2, strides=2, padding='same')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=192, kernel_size=(5,5), activation='relu')(x)
    x = Conv2D(filters=192, kernel_size=(1,1), activation='relu')(x)
    x = Conv2D(filters=192, kernel_size=(1,1), activation='relu')(x)
    x = MaxPool2D(2, strides=2, padding='same')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=192, kernel_size=(3,3), activation='relu')(x)
    x = Conv2D(filters=192, kernel_size=(1,1), activation='relu')(x)
    x = Conv2D(filters=10, kernel_size=(1,1), activation='relu')(x)

    output = GlobalAveragePooling2D()(x)

    model = Model(input, output)

    return model 


if __name__ == '__main__':
    
    model = nin(input_shape=(227,227,3))

    # Show the full model structure
    model.summary()
