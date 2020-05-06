from tensorflow.keras import layers

def Decoding(x):
    """
    ResNeXt by default. For ResNet set `cardinality` = 1 above.

    """

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1)):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        # y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        # y = layers.LeakyReLU()(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        # y = layers.BatchNormalization()(y)
        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        # y = layers.LeakyReLU()(y)
        return y

    # conv1
    for i in range(8):
        x = residual_block(x, 64, 64)
    # conv1
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)

    return x

