from tensorflow.keras import layers

def Decoding(x):


    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1)):

        shortcut = y

        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        # y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        # y = layers.LeakyReLU()(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        # y = layers.BatchNormalization()(y)
        y = layers.add([shortcut, y])

        # y = layers.LeakyReLU()(y)
        return y

    # conv1
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for i in range(8):
        x = residual_block(x, 64, 64)
    # conv1
    x = layers.Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)

    return x

