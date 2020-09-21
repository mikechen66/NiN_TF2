
# nin.py

"""
NIN(Network in Network) is the creative DNN model in which GAN(Global Averag Pooling) is adpted to elimitate the 
large quantity of parameters. It deals with 10 label classes.  In contrast, AlexNet deep learning structure is a 
combination of CNN+FC that incurs a huge RAM occupation. CNN is responsible for extracting features, and FC is 
responsible for feature classification. NIN uses mlpconv and GAP to organicall combine the two parts of CNN and 
streamline FC with making it more interpretable. It is a function-style realization. 

Reference:
https://arxiv.org/pdf/1312.4400.pdf
"""


# -import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Dropout, Activation, MaxPooling2D, GlobalAveragePooling2D
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomNormal  
from keras.regularizers import l2


def nin(input_shape):

    input = Input(shape=input_shape)

    x = Conv2D(filters=192, kernel_size=(5,5), padding='same', activation="relu", kernel_regularizer=l2(1e-4))(input)
    x = BatchNormalization()(x)
    x = Conv2D(filters=160, kernel_size=(1,1), padding='same', activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=96, kernel_size=(1,1), padding='same', activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=192, kernel_size=(5,5), padding='same', activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=192, kernel_size=(1,1), padding='same', activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=192, kernel_size=(1,1), padding='same', activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=192, kernel_size=(3,3), padding='same', activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=192, kernel_size=(1,1), padding='same', activation="relu" , kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=10, kernel_size=(1,1), padding='same', activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    output = Activation(activation="softmax")(x)

    model = Model(input, output)

    return model


if __name__ == '__main__':

    image_width = 227
    image_height = 227
    channels = 3

    # Assign the values 
    input_shape = (image_width, image_height, channels)

    model = nin(input_shape)

    # show the full model structure
    model.summary()

