#!/usr/bin/env python
# coding: utf-8

# nin_rec.py

"""
The script realizes the stack of the layers with the complete recursive function 
style. It is better to define an outer function to include the common arguments 
and return "layer", the name of inner function of layer(). And then define an 
inner function that is recursively called by get_mode(). Otherwise, it is necessary 
to include the formal argument b1,b2,b3 in statements of nin_block() within the 
function defintion of get_model. 
"""


from keras.layers import Input, Conv2D, Dropout, Dense, Activation, MaxPooling2D, \
    AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model


def nin_block(kernel, levels, strides):

    def layer(x):
        # Put the arguyment "array" in the front of kernel since it is a list object. 
        a = Conv2D(levels[0], kernel, strides=strides, padding='same')(x)
        a = Activation('relu')(a)
        for size in levels[1:]:
            a = Conv2D(size, 1, strides=(1,1))(a)
            a = Activation('relu')(a)

        return a

    return layer


def get_model(img_rows, img_cols):
    # Input() initizate a 3-D shape(w,h,c) into a 4-D tensor(b,w,h,c). 
    img = Input(shape=(img_rows, img_cols, 3))
    b1 = nin_block(kernel=(5,5), levels=[192,160,96], strides=(1,1))(img)
    b1 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(b1)
    b1 = Dropout(0.5)(b1)

    b2 = nin_block(kernel=(5,5), levels=[192,192,192], strides=(1,1))(b1)
    b2 = AveragePooling2D(pool_size=(3, 3), strides=(2,2))(b2)
    b2 = Dropout(0.5)(b2)

    b3 = nin_block(kernel=(3,3), levels=[192,192,10], strides=(1,1))(b2)

    b4 = GlobalAveragePooling2D()(b3)
    b4 = Activation('softmax')(b4)

    model = Model(inputs=img, outputs=b4)

    return model


if __name__ == '__main__':

    model = get_model(227, 227)

    model.summary()