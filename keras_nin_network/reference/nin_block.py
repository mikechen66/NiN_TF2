#!/usr/bin/env python
# coding: utf-8

# nin_rec.py

"""
The script realizes the stack of the layers with the complete assignmentfunction 
style. It is better to define an outer function to include the common arguments 
and return "layer", the name of inner function of layer(). And then define an 
inner function that is called by get_mode(). Otherwise, it is necessary to include 
b1,b2,b3 in assignment statements of nin_block() within the function defintion of 
get_model. 
"""


from keras.layers import Input, Conv2D, Dropout, Dense, Activation, MaxPooling2D, \
    AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model


def block(kernel_size, filters, strides):

    def layer(x):
        # Put the arguyment "array" in the front of kernel since it is a list object. 
        a = Conv2D(filters[0], kernel_size, strides=strides, padding='same')(x)
        a = Activation('relu')(a)
        for number in filters[1:]:
            a = Conv2D(number, 1, strides=(1,1))(a)
            a = Activation('relu')(a)

        return a

    return layer


def call(input_shape):
    # Input() initizate a 3-D shape(w,h,c) into a 4-D tensor(b,w,h,c). 
    img = Input(shape=input_shape)
    b1 = block(kernel_size=(5,5), filters=[192,160,96], strides=(1,1))(img)
    b1 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(b1)
    b1 = Dropout(0.5)(b1)

    b2 = block(kernel_size=(5,5), filters=[192,192,192], strides=(1,1))(b1)
    b2 = AveragePooling2D(pool_size=(3, 3), strides=(2,2))(b2)
    b2 = Dropout(0.5)(b2)

    b3 = block(kernel_size=(3,3), filters=[192,192,10], strides=(1,1))(b2)

    b4 = GlobalAveragePooling2D()(b3)
    b4 = Activation('softmax')(b4)

    model = Model(img, b4)

    return model


if __name__ == '__main__':

    model = call(input_shape=(227,227,3))

    model.summary()