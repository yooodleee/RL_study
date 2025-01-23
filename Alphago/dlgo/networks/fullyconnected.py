from keras import layers    # Dense, Activation, Flatten


def layers(input_shape):
    return [
        layers.Dense(128, input_shape=input_shape),
        layers.Activation('relu'),
        layers.Dense(128, input_shape=input_shape),
        layers.Activation('relu'),
        layers.Flatten(),
        layers.Dense(128),
        layers.Activation('relu'),
    ]