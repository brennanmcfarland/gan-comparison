from tensorflow.keras.layers import Input, Conv2D, Conv3D, BatchNormalization, LeakyReLU, Flatten, Dense, Activation, Reshape, PReLU, Conv2DTranspose, Dropout
from layers import *


def basic_submodel():

    d_input = Input(shape=(28, 28, 1))
    layer = Conv2D(filters=64, kernel_size=(4, 4), padding='same')(d_input)
    layer = BatchNormalization()(layer)
    layer = SpectralNorm()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = Dropout(.3)(layer)
    layer = Conv2D(filters=64, kernel_size=(4, 4), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = SpectralNorm()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = Dropout(.3)(layer)
    layer = Conv2D(filters=32, kernel_size=(4, 4), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = SpectralNorm()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = MiniBatchStdDev()(layer)
    layer = Flatten()(layer)
    layer = Dense(128)(layer)
    layer = SpectralNorm()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = Dense(1)(layer)
    layer = Activation('sigmoid')(layer)
    d_output = layer

    g_input = Input(shape=(100,))
    layer = Dense(2048)(g_input)
    layer = Dense(784)(layer)
    layer = Reshape((28, 28, 1))(layer)
    layer = Conv2DTranspose(128, 4, strides=1, padding='same')(layer)
    layer = BatchNormalization(momentum=.9)(layer)
    layer = PReLU()(layer)
    for _ in range(4):
        layer = Conv2DTranspose(128, 4, strides=1, padding='same')(layer)
        layer = BatchNormalization(momentum=.9)(layer)
        layer = PReLU()(layer)
    layer = Conv2DTranspose(1, 3, strides=1, padding='same')(layer)
    g_output = Activation('sigmoid')(layer)
    return g_input, g_output, d_input, d_output


# TODO: the reason it's erroring out is because Conv2D expects (batch, x, y, channel) and not the extra image dimension
# TODO: at the end; could solve either with 3d convolution (keeping the kernel depth small since there's not much deptwise)
# TODO: or by making a custom layer that splits the input tensor along the image axis, convolves along each separately
# TODO: and then merges them all together again; probably try both and see which one works better
def choicegan_submodel(num_choices=2):
    d_input = Input(shape=(28, 28, num_choices, 1))
    layer = Conv3D(filters=64, kernel_size=(4, 4, num_choices), padding='same')(d_input)
    layer = BatchNormalization()(layer)
    layer = SpectralNorm()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = Dropout(.3)(layer)
    layer = Conv3D(filters=64, kernel_size=(4, 4, num_choices), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = SpectralNorm()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = Dropout(.3)(layer)
    layer = Conv3D(filters=32, kernel_size=(4, 4, num_choices), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = SpectralNorm()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = MiniBatchStdDev()(layer)
    layer = Flatten()(layer)
    layer = Dense(128)(layer)
    layer = SpectralNorm()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = Dense(1)(layer)
    layer = Activation('sigmoid')(layer)
    d_output = layer

    g_input = Input(shape=(100,))
    layer = Dense(2048)(g_input)
    layer = Dense(784)(layer)
    layer = Reshape((28, 28, 1))(layer)
    layer = Conv2DTranspose(128, 4, strides=1, padding='same')(layer)
    layer = BatchNormalization(momentum=.9)(layer)
    layer = PReLU()(layer)
    for _ in range(4):
        layer = Conv2DTranspose(128, 4, strides=1, padding='same')(layer)
        layer = BatchNormalization(momentum=.9)(layer)
        layer = PReLU()(layer)
    layer = Conv2DTranspose(1, 3, strides=1, padding='same')(layer)
    g_output = Activation('sigmoid')(layer)
    return g_input, g_output, d_input, d_output


def autoencoder_submodel(num_choices=2, num_d_channels=1):
    d_input = Input(shape=(28, 28, num_choices, num_d_channels))
    layer = Conv3D(filters=64, kernel_size=(4, 4, num_choices), padding='same')(d_input)
    layer = BatchNormalization()(layer)
    layer = SpectralNorm()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = Dropout(.3)(layer)
    layer = Conv3D(filters=64, kernel_size=(4, 4, num_choices), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = SpectralNorm()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = Dropout(.3)(layer)
    layer = Conv3D(filters=32, kernel_size=(4, 4, num_choices), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = SpectralNorm()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = MiniBatchStdDev()(layer)
    layer = Flatten()(layer)
    layer = Dense(128)(layer)
    layer = SpectralNorm()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = Dense(1)(layer)
    layer = Activation('sigmoid')(layer)
    d_output = layer

    # the generator is the decoder for the autoencoder
    g_input = Input(shape=(100,))
    layer = Dense(2048)(g_input)
    layer = Dense(784)(layer)
    layer = Reshape((28, 28, 1))(layer)
    layer = Conv2DTranspose(128, 4, strides=1, padding='same')(layer)
    layer = BatchNormalization(momentum=.9)(layer)
    layer = PReLU()(layer)
    for _ in range(4):
        layer = Conv2DTranspose(128, 4, strides=1, padding='same')(layer)
        layer = BatchNormalization(momentum=.9)(layer)
        layer = PReLU()(layer)
    layer = Conv2DTranspose(1, 3, strides=1, padding='same')(layer)
    g_output = Activation('sigmoid')(layer)

    # encoder for the autoencoder
    e_input = Input(shape=(28, 28, 1))
    layer = Conv2D(filters=64, kernel_size=(4, 4), padding='same')(e_input)
    layer = BatchNormalization()(layer)
    layer = SpectralNorm()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = Dropout(.3)(layer)
    for _ in range(4):
        layer = Conv2D(filters=64, kernel_size=(4, 4), padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = SpectralNorm()(layer)
        layer = LeakyReLU(.25)(layer)
        layer = Dropout(.3)(layer)
    layer = Flatten()(layer)
    layer = Dense(256)(layer)
    layer = SpectralNorm()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = Dense(128)(layer)
    layer = SpectralNorm()(layer)
    layer = LeakyReLU(.25)(layer)
    layer = Dense(100)(layer)
    # TODO: part of the reason just prepending the encoder seems to be changing/improving it so much might be due to
    # TODO: the fact that the linear activation layer means the inputs to the decoder are stored in a pattern that is
    # TODO: easier to generate from and that's numerically biasing it towards far greater diversity, investigate?
    # TODO: it's not just the range of input values; tried increasing it and making half negative to no effect
    # TODO: could just be the additional layers are giving it a greater solution space to roam making it take longer to
    # TODO: converge/making gradient normalization necessary but possibly improving results in the long run (test it)
    layer = Activation('linear')(layer)
    e_output = layer

    return g_input, g_output, e_input, e_output, d_input, d_output


def distributional_autoencoder_submodel(num_choices=2):
    return autoencoder_submodel(num_choices, 101)