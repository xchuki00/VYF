from tensorflow.keras import layers

#
# network params
#

cardinality = 32


def Encoding(x):


    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1)):

        shortcut = y

        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=_strides, padding='same',trainable=False)(y)
        # y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        # y = layers.LeakyReLU()(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(3, 3), strides=_strides, padding='same',trainable=False)(y)
        # y = layers.BatchNormalization()(y)
        y = layers.add([shortcut, y])

        # y = layers.LeakyReLU()(y)
        return y

    # conv1
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',trainable=False)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # 2x res
    for i in range(2):
        x = residual_block(x, 64, 64)
    # 2x down
    es1 = x

    x = layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same',trainable=True)(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same',trainable=True)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    es2 = x

    x = layers.Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same',trainable=True)(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same',trainable=True)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # 4x res
    for i in range(4):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        x = residual_block(x, 256, 256, _strides=(1,1))
    # 2x up
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same',trainable=True)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.add([es2, x])

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',trainable=True)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.add([es1, x])

    # 2x res
    for i in range(2):
        x = residual_block(x, 64, 64)


    # conv1
    x = layers.Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same',trainable=False)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)

    return x

