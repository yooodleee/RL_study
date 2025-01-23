from keras import layers    # Dense, Activation, Flatten(core)
from keras import layers    # Conv2D, ZeroPadding2D(convolutional)


def layer(input_shape):
    return [
        layers.ZeroPadding2D(padding=3, input_shape=input_shape, data_format='channels_first'),
        layers.Conv2D(48, (7, 7), data_format='channels_first'),
        layers.Activation('relu'),

        layers.ZeroPadding2D(padding=2, data_format='channels_first'),
        layers.Conv2D(32, (5, 5), data_format='channels_first'),
        layers.Activation('relu'),

        layers.ZeroPadding2D(padding=2, data_format='channels_first'),
        layers.Conv2D(32, (5, 5), data_format='channels_first'),
        layers.Activation('relu'),

        layers.ZeroPadding2D(padding=2, data_format='channels_first'),
        layers.Conv2D(32, (5, 5), data_format='channels_first'),
        layers.Activation('relu'),

        layers.Flatten(),
        layers.Dense(512),
        layers.Activation('relu'),
    ]