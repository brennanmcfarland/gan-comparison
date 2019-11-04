from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dropout, \
    Activation, GaussianNoise, Reshape, Add, Flatten, LeakyReLU, Input
from tensorflow.keras.constraints import min_max_norm
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import numpy as np
import tensorflow as tf


# TODO: clean up this mess
class ToChoices(Layer):
    def call(self, inputs): # reals, fakes, shuffle indices
        reals, fakes, shuffle_indices = inputs
        choice_images = tf.stack((fakes, reals), axis=-2)
        # TODO: the problem is that shuffle_indices is of shape batch size, what we need is an index for choice rather than batch item
        # TODO: so shuffle_indices should be 2D, one axis for which one in batch and one for the ordering of choices
        # TODO: shuffle_indices is now indexed by batch then by ordering of choices, so now we just need to get it to
        # TODO: shuffle along the right axis
        shuffle_indices = tf.cast(shuffle_indices, dtype=tf.int64)
        #choice_images = tf.gather(choice_images, shuffle_indices, axis=3)
        #choice_images = tf.map_fn(, tf.stack((choice_images, shuffle_indices)
        def apply_gather(x):
            indices = x[1,0,0,:,0]
            return tf.gather(x, tf.cast(indices, tf.int32), axis=-2) # TODO: something's not right here based on https://stackoverflow.com/questions/55597335/how-to-use-tf-gather-in-batch
        # TODO: output dimension is clearly messed up
        choice_images = tf.cast(choice_images, dtype=tf.float32)
        shuffle_indices = tf.cast(shuffle_indices, dtype=tf.float32)
        #expanded_shuffle_indices = tf.identity(choice_images) #tf.zeros(shape=choice_images.shape)
        #expanded_shuffle_indices = tf.map_fn(lambda i: tf.fill(dims=expanded_shuffle_indices.shape[1:], value=i[0]), shuffle_indices)
        for i in fakes.shape[1:-1]:
            shuffle_indices = tf.stack([shuffle_indices for _ in range(i)], axis=1)#tf.tile(shuffle_indices, tf.expand_dims(i, axis=0))
        shuffle_indices = tf.stack([shuffle_indices for _ in range(fakes.shape[-1])], axis=-1)
        # for _ in range(2):
        #     shuffle_indices = tf.expand_dims(shuffle_indices, axis=-1)
        stacked = tf.stack([choice_images, shuffle_indices], axis=1)
        #stacked = tf.stack([choice_images, expanded_shuffle_indices], axis=1)
        # TODO: it's working great up to here and indices are stacked with choices on first non-batch axis, so just pick
        # TODO: up from here, try to rewrite the apply_gather function to gather the choices in the order given by the indices
        stacked = tf.ensure_shape(stacked, shape=(32, 2, 28, 28, 2, 1))
        gathered = tf.map_fn(apply_gather, stacked)
        gathered = gathered[:,0]
        choice_images = tf.stack(gathered, axis=0)

        #choice_images = tf.squeeze(choice_images, axis=-2)
        #choice_images = tf.ensure_shape(choice_images, shape=self.compute_output_shape((i.shape for i in inputs))) # TODO: uncomment
        return choice_images

    # output shape is verified to be correct
    def compute_output_shape(self, input_shapes): # reals, fakes, shuffle indices
        reals_shape, fakes_shape, _ = input_shapes
        #output_shape = tf.concat((reals_shape[:-1], reals_shape[-1] + fakes_shape[-1]), axis=0)
        output_shape = reals_shape[:-1].concatenate(reals_shape[-1] + fakes_shape[-1]).concatenate(reals_shape[-1])
        return output_shape

    # TODO: this sucks
    def ensure_batch_size(self, input):
        return tf.concat(tf.constant((32,)), (input.shape[1:]))


# TODO: just copy pasted this from online make it nice, and maybe we can use it
class SpectralNorm(Layer):
    def build(self, input_shape):
        self.u = self.add_weight(
            name='u',
            shape=tf.stack((1, input_shape[-1])),
            initializer=tf.random_normal_initializer(),
            trainable=False
        )
        self.built = True

    def call(self, inputs):
        iteration = 1
        w_shape = inputs.shape
        inputs = tf.reshape(inputs, [-1, w_shape[-1]])
        w = inputs
        u_hat = self.u
        v_hat = None
        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([self.u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, tf.pad(w_shape[1:], [[1, 0]], constant_values=-1))

        return w_norm

class EqualizedDense(Dense):
    def call(self, inputs):
        output = K.dot(inputs, scale_weights(self.kernel))
        if self.use_bias:
            output = K.bias_add(output, scale_weights(self.bias), data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


class EqualizedConv2D(Conv2D):
    def call(self, inputs):
        outputs = K.conv2d(
            inputs,
            scale_weights(self.kernel),
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                scale_weights(self.bias),
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class EqualizedLayer(Layer):

    def __new__(cls, sublayer, **kwargs):
        # intercept and modify build call?
        old_build_f = sublayer.build
        def new_f(obj, old_f, input_shape):
            old_f(input_shape)
            obj.set_weights([
                property(
                    lambda self: scale_weights(self),
                    lambda self, value: setattr(self, self.__name, value)
                ) for w in sublayer.get_weights()])
        new_build_f = lambda input_shape: new_f(sublayer, old_build_f, input_shape)
        sublayer.build = new_build_f
        return sublayer


class PixelNorm(Layer):
    def __init__(self, eps=1e-8, **kwargs):
        self.eps = eps
        super(PixelNorm, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        with tf.variable_scope('PixelNorm'):
            return x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keep_dims=True) + self.eps)


class MiniBatchStdDev(Layer):
    def __init__(self, **kwargs):
        super(MiniBatchStdDev, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + (input_shape[-1] + 1,)

    def call(self, x):
        std = tf.keras.backend.std(x, axis=0)
        std = tf.math.reduce_mean(std)
        std_shape = tf.shape(tf.expand_dims(tf.unstack(x, axis=-1)[0], axis=-1))
        constant_std_tensor = tf.fill(std_shape, std)
        output_tensor = tf.concat([x, constant_std_tensor], axis=-1)
        return output_tensor


class AdaIN(Layer):
    def __init__(self, data_format=None, eps=1e-7, **kwargs):
        super(AdaIN, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.spatial_axis = [1, 2] if self.data_format == 'channels_last' else [2, 3]
        self.eps = eps

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs):
        image = inputs[0]
        if len(inputs) == 2:
            style = inputs[1]
            style_mean, style_var = tf.nn.moments(style, self.spatial_axis, keep_dims=True)
        else:
            style_mean = tf.expand_dims(K.expand_dims(inputs[1], self.spatial_axis[0]), self.spatial_axis[1])
            style_var = tf.expand_dims(K.expand_dims(inputs[2], self.spatial_axis[0]), self.spatial_axis[1])
        image_mean, image_var = tf.nn.moments(image, self.spatial_axis, keep_dims=True)
        out = tf.nn.batch_normalization(image, image_mean,
                                        image_var, style_mean,
                                        tf.sqrt(style_var), self.eps)
        return out


class NoiseInput(Layer):
    def __init__(self, shape, **kwargs):
        self.shape = shape
        super(NoiseInput, self).__init__(**kwargs)

    def compute_output_shape(self, _):
        return self.shape

    def build(self, _):
        self.means = self.add_weight(name='means',
                                     shape=self.shape[1:],  # exclude batch
                                     initializer='random_normal',
                                     trainable=True)
        self.variances = self.add_weight(name='variances',
                                         shape=self.shape[1:],
                                         initializer='random_normal',
                                         trainable=True)
        super(NoiseInput, self).build(self.shape)  # Be sure to call this at the end

    def call(self, inputs):
        return K.random_normal(shape=K.shape(inputs),
                               mean=0.,
                               stddev=self.variances) + self.means

    # necessary for proper serialization
    def get_config(self):
        config = {'shape': self.shape}
        base_config = super(NoiseInput, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EqualizedNoiseInput(NoiseInput):
    def call(self, inputs):
        return K.random_normal(shape=K.shape(inputs),
                               mean=0.,
                               stddev=scale_weights(self.variances)) + scale_weights(self.means)


class ScaleLayer(Layer):
    def __init__(self, scale_factor, **kwargs):
        self.scale_factor = scale_factor
        super(ScaleLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        return tf.math.scalar_mul(self.scale_factor, x)


def normalize_data_format(value):
    if value is None:
        value = 'channels_last'
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format


def scale_weights(w):
    return w * tf.sqrt(2.0 / tf.size(w, out_type=tf.float32))