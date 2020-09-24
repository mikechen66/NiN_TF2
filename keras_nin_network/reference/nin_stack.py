#!/usr/bin/env python
# coding: utf-8

# nin_stack.py

"""
The script realizes the stack of layers with function call. It includes a function 
call with consecutive assignment statments. For instance, it is necessary to include 
"img" in statements of block() within the function defintion of call. It is much more
complex than the recrusive realization. 
"""


from keras.layers import Input, Conv2D, Dropout, Dense, Activation, MaxPooling2D, \
    AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model


def block(x, kernel, levels, strides):
    # Put the arguyment "array" in the front of kernel since it is a list object.
    a = Conv2D(levels[0], kernel, strides=strides, padding='same')(x)
    a = Activation('relu')(a)
    for size in levels[1:]:
        a = Conv2D(size, 1, strides=(1,1))(a)
        a = Activation('relu')(a)

    return a


def call(input_shape):

    # Input() initizate a 3-D shape(w,h,c) into a 4-D tensor(b,w,h,c). 
    img = Input(shape=input_shape)
    b1  = block(img, kernel=(5,5), levels=[192,160,96], strides=(1,1))
    b1  = MaxPooling2D(pool_size=(3,3), strides=(2,2))(b1)
    b1  = Dropout(0.5)(b1)

    b2  = block(b1, kernel=(5,5), levels=[192,192,192], strides=(1,1))
    b2  = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(b2)
    b2  = Dropout(0.5)(b2)

    b3  = block(b2, kernel=(3,3), levels=[192,192,10], strides=(1,1))

    b4  = GlobalAveragePooling2D()(b3)
    b4  = Activation('softmax')(b4)

    model = Model(inputs=img, outputs=b4)

    return model


if __name__ == '__main__':

    model = call(input_shape=(227,227,3))

    model.summary()