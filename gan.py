from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import initializers
from tensorflow.keras.layers import Input
import tensorflow as tf
from tensorflow.keras.layers import Concatenate
import numpy as np
from skimage.transform import downscale_local_mean
from skimage import io
import random
from layers import ToChoices, ConcatWithEncoded
from tensorflow.keras.utils import plot_model


output_root = './output/'


def report_epoch_progress(epoch, logs, gen_model, data_test):
    if epoch is None:
        epoch = "initial"
    print('epoch', epoch, 'progress report:')
    example = data_test.__getitem__(3)
    latent = example[0]
    img = gen_model.predict(latent)
    img = np.squeeze(img, axis=3)
    img = np.concatenate(img)
    img *= 255.0
    img = img.astype(int)

    actual = example[1]
    actual = np.squeeze(actual, axis=3)
    actual = np.concatenate(actual)
    actual *= 255.0
    actual = actual.astype(int)

    print("OUTPUT")
    # io.imshow(img, cmap='gray')
    # io.show()
    output_name = output_root + "output" + str(epoch) + ".png"
    io.imsave(output_name, img)
    print("TARGET")
    target_name = output_root + "target" + str(epoch) + ".png"
    io.imsave(target_name, actual)
    # io.imshow(actual, cmap='gray')
    # io.show()


class BasicGAN():

    def __init__(self, g, d):
        self.generator = g
        self.discriminator = d
        self.num_data = 0
        self.batch_size = 32

        gan_inputs = Input(shape=(100,))
        gan_images = self.generator(gan_inputs)
        gan_output = self.discriminator(gan_images)

        self.gan = Model(inputs=gan_inputs, outputs=[gan_images, gan_output])

    def compile(self):
        d_optimizer = Adam(
            lr=.0001)  # lower lr necessary to keep discriminator from getting too much better than generator
        gan_optimizer = Adam(lr=.0001)

        self.discriminator.trainable = True
        self.discriminator.compile(optimizer=d_optimizer, loss='binary_crossentropy')
        self.discriminator.trainable = False
        self.gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
        self.discriminator.trainable = True

        # only the progress callback gets used right now, as I'm not sure how to make this work with a custom training loop
        progress_callback = LambdaCallback(on_epoch_end=report_epoch_progress)
        checkpoint_callback = ModelCheckpoint('./model-checkpoint.ckpt')
        tensorboard_callback = TensorBoard(log_dir='../logs/tensorboard-logs', write_images=True)
        self.callbacks = [progress_callback]
        self.gan.summary()

    def train(self, epochs, data_train, data_test, verbose=False, pretraining_steps = 0):
        self.data_train = data_train
        self.data_test = data_test
        steps_per_epoch = self.num_data // self.data_train.batch_size
        discriminator_updates = 2  # updates per discriminator update
        report_epoch_progress(None, None, self.generator, self.data_test)

        # manually set callbacks since we're doing a custom fit
        for callback in self.callbacks:
            callback.set_model(self.gan)

        for step in range(pretraining_steps):
            current_batch = self.data_train.__getitem__(0)
            # generate "fake" images
            generator_inputs = current_batch[0]
            generated_images = self.generator.predict(generator_inputs, batch_size=self.batch_size)
            # train discriminator on "real" images from the dataset and "fake" ones that were generated
            true_inputs, true_labels = current_batch[1], np.random.uniform(.9, 1.0, size=(len(current_batch[1]),))
            fake_inputs, fake_labels = generated_images, np.random.uniform(0.0, .1, size=(len(generated_images),))
            d_loss_real = self.discriminator.train_on_batch(true_inputs, true_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_inputs, fake_labels)
            d_loss = .5 * np.sum((d_loss_real, d_loss_fake))
            if verbose == True:
                print('PRETRAINING step ', step, '/', pretraining_steps, 'd_loss: ', d_loss)

        # custom training loop
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                current_batch = self.data_train.__getitem__(0)
                # generate "fake" images
                generator_inputs = current_batch[0]
                generated_images = self.generator.predict(generator_inputs, batch_size=self.batch_size)
                # train discriminator on "real" images from the dataset and "fake" ones that were generated
                true_inputs, true_labels = current_batch[1], np.random.uniform(.9, 1.0, size=(len(current_batch[1]),))
                fake_inputs, fake_labels = generated_images, np.random.uniform(0.0, .1, size=(len(generated_images),))
                if step % discriminator_updates == 0:
                    d_loss_real = self.discriminator.train_on_batch(true_inputs, true_labels)
                    d_loss_fake = self.discriminator.train_on_batch(fake_inputs, fake_labels)
                    d_loss = .5 * np.sum((d_loss_real, d_loss_fake))
                # halt training on the discriminator
                self.discriminator.trainable = False
                # train generator to try to fool the discriminator
                g_loss = self.gan.train_on_batch(generator_inputs, [true_inputs, true_labels])
                # and allow training on the discriminator to continue
                self.discriminator.trainable = True
                if verbose == True:
                    print('step ', step, '/', steps_per_epoch, 'd_loss: ', d_loss, 'g_loss: ', g_loss)
            # manually call callbacks since we're doing a custom fit
            logs = {'loss': g_loss}
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs, self.generator, self.data_test)
        # manually terminate callbacks since we're doing a custom fit
        for callback in self.callbacks:
            callback.on_train_end('_')


# TODO: is there any way to optimize this?  why is it so sloooow?
# basically just trains on the real and fake at once, so not much different from the regular gan
# TODO: have multiple images in a batch (can add at end of shape) and force it to choose between them (probably just 2 to start, but try more also)
class BatchChoiceGAN():
    def __init__(self, g, d):
        self.generator = g
        self.discriminator = d
        self.num_data = 0
        self.batch_size=32

        gan_inputs = Input(shape=(100,))
        gan_images = self.generator(gan_inputs)
        gan_output = self.discriminator(gan_images)

        self.gan = Model(inputs=gan_inputs, outputs=[gan_images, gan_output])

    def compile(self):
        d_optimizer = Adam(
            lr=.00001, beta_1=.5)  # lower lr necessary to keep discriminator from getting too much better than generator
        gan_optimizer = Adam(lr=.0001, beta_1=.5)

        self.discriminator.trainable = True
        self.discriminator.compile(optimizer=d_optimizer, loss='binary_crossentropy')
        self.discriminator.trainable = False
        loss = ['binary_crossentropy', 'mse']
        loss_weights = [100, 1]
        self.gan.compile(optimizer=gan_optimizer, loss=loss, loss_weights=loss_weights)
        self.discriminator.trainable = True

        # only the progress callback gets used right now, as I'm not sure how to make this work with a custom training loop
        progress_callback = LambdaCallback(on_epoch_end=report_epoch_progress)
        checkpoint_callback = ModelCheckpoint('./model-checkpoint.ckpt')
        tensorboard_callback = TensorBoard(log_dir='../logs/tensorboard-logs', write_images=True)
        self.callbacks = [progress_callback]
        self.gan.summary()

    # generate a list of images, half real, half fake, with the correct labels to train the discriminator
    # TODO: try with and without encoder also
    def generate_choices(self):
        current_batch = self.data_train.__getitem__(0)
        # generate "fake" images
        generator_inputs = current_batch[0]
        generated_images = self.generator.predict(generator_inputs, batch_size=self.batch_size)
        target_images = current_batch[1]
        choice_images = tf.concat((generated_images, target_images), axis=0) # merge the lists of real and fake images
        choice_labels = tf.cast(tf.concat(
            (
                np.random.uniform(0.0, .1, size=(len(generated_images),),),
                np.random.uniform(.9, 1.0, size=(len(current_batch[1]),))
            ), axis=0), tf.float32) # merge the lists of real and fake labels

        shuffle_indices = tf.random.shuffle(tf.range(self.batch_size))
        choice_images = tf.gather(choice_images, shuffle_indices, axis=0)
        choice_labels = tf.gather(choice_labels, shuffle_indices, axis=0)
        return choice_images, choice_labels, generator_inputs, generated_images, current_batch[1] # images, labels, fake latents, fake images, true images

    def train(self, epochs, data_train, data_test, verbose=False, pretraining_steps = 0):
        self.data_train = data_train
        self.data_test = data_test
        steps_per_epoch = self.num_data // self.data_train.batch_size
        discriminator_updates = 2  # updates per discriminator update
        report_epoch_progress(None, None, self.generator, self.data_test)

        # manually set callbacks since we're doing a custom fit
        for callback in self.callbacks:
            callback.set_model(self.gan)

        for step in range(pretraining_steps):
            choice_images, choice_labels, _, _, _ = self.generate_choices()
            d_loss = self.discriminator.train_on_batch(choice_images, choice_labels)
            if verbose == True:
                print('PRETRAINING step ', step, '/', pretraining_steps, 'd_loss: ', d_loss)

        # custom training loop
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                choice_images, choice_labels, fake_latents, fake_images, true_images = self.generate_choices() # TODO: may be more efficient to split into real and fake generation and merging, since each is only used sometimes
                if step % discriminator_updates == 0:
                    d_loss = self.discriminator.train_on_batch(choice_images, choice_labels)
                # halt training on the discriminator
                self.discriminator.trainable = False
                # train generator to try to fool the discriminator
                g_loss = self.gan.train_on_batch(fake_latents, [true_images, np.random.uniform(.9, 1.0, size=(len(true_images),))])
                # and allow training on the discriminator to continue
                self.discriminator.trainable = True
                if verbose == True:
                    print('step ', step, '/', steps_per_epoch, 'd_loss: ', d_loss, 'g_loss: ', g_loss)
            # manually call callbacks since we're doing a custom fit
            logs = {'loss': g_loss}
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs, self.generator, self.data_test)
        # manually terminate callbacks since we're doing a custom fit
        for callback in self.callbacks:
            callback.on_train_end('_')


# have multiple images in a batch (can add at end of shape) and force it to choose between them
class ChoiceGAN():
    def __init__(self, g, d, num_choices=2):
        self.generator = g
        self.discriminator = d
        self.num_data = 0
        self.batch_size=32
        self.num_choices = num_choices

        gan_inputs = Input(shape=(100,))
        gan_images = self.generator(gan_inputs)
        real_inputs = Input(shape=(28, 28, 1))
        shuffle_indices = Input(shape=(num_choices,)) # batch shape not included
        gan_choices = ToChoices()([real_inputs, gan_images, shuffle_indices])
        ToChoices().compute_output_shape([real_inputs.shape, gan_images.shape, shuffle_indices.shape])
        ToChoices().call([real_inputs, gan_images, shuffle_indices]) # TODO: remove this, for debugging
        self.discriminator.summary()
        gan_output = self.discriminator(gan_choices)
        self.generator.summary()

        self.gan = Model(inputs=[gan_inputs, real_inputs, shuffle_indices], outputs=[gan_images, gan_output])

    def compile(self):
        d_optimizer = Adam(
            lr=.00001, beta_1=.5)  # lower lr necessary to keep discriminator from getting too much better than generator
        gan_optimizer = Adam(lr=.0001, beta_1=.5)

        self.discriminator.trainable = True
        self.discriminator.compile(optimizer=d_optimizer, loss='binary_crossentropy')
        self.discriminator.trainable = False
        loss = ['binary_crossentropy', 'mse']
        loss_weights = [100, 1]
        self.gan.compile(optimizer=gan_optimizer, loss=loss, loss_weights=loss_weights)
        self.discriminator.trainable = True

        # only the progress callback gets used right now, as I'm not sure how to make this work with a custom training loop
        progress_callback = LambdaCallback(on_epoch_end=report_epoch_progress)
        checkpoint_callback = ModelCheckpoint('./model-checkpoint.ckpt')
        tensorboard_callback = TensorBoard(log_dir='../logs/tensorboard-logs', write_images=True)
        self.callbacks = [progress_callback]
        self.gan.summary()

    # generate a list of images, half real, half fake, with the correct labels to train the discriminator
    # TODO: cleanup and convert numpy stuff to tf
    # image dims are (batch, x, y, channels, image)
    # label dims should be (batch, image)
    def generate_choices(self):
        # TODO: in the current implementation, should only work if the number of choices is even, but that's probably ok
        current_batch = [self.data_train.__getitem__(0) for _ in range(self.num_choices//2)]
        # generate "fake" images
        generator_inputs = np.stack([b[0] for b in current_batch], axis=-1)
        generated_images = np.stack([
            self.generator.predict(np.squeeze(gen_input), batch_size=self.batch_size) for gen_input in
            np.split(generator_inputs, generator_inputs.shape[-1], axis=-1)
        ], axis=-1)
        target_images = np.stack([b[1] for b in current_batch], axis=-1)
        #target_images = [b[1] for b in current_batch]
        #target_images = np.array(target_images)
        choice_images = tf.concat((generated_images, target_images), axis=-2) # merge the lists of real and fake images
        choice_labels = tf.cast(tf.stack(
            (
                np.random.uniform(0.0, .1, size=(len(generated_images),),),
                np.random.uniform(.9, 1.0, size=(len(current_batch[0][1]),))
            ), axis=-1), tf.float32) # merge the lists of real and fake labels
        shuffle_indices = tf.stack([tf.random.shuffle(tf.range(self.num_choices)) for _ in range(self.batch_size)], axis=0)

        choice_images = tf.cast(choice_images, dtype=tf.float32)
        shuffle_indices = tf.cast(shuffle_indices, dtype=tf.float32)
        # expanded_shuffle_indices = tf.identity(choice_images) #tf.zeros(shape=choice_images.shape)
        # expanded_shuffle_indices = tf.map_fn(lambda i: tf.fill(dims=expanded_shuffle_indices.shape[1:], value=i[0]), shuffle_indices)
        for i in generated_images.shape[1:-2]:
            shuffle_indices = tf.stack([shuffle_indices for _ in range(i)],
                                       axis=1)  # tf.tile(shuffle_indices, tf.expand_dims(i, axis=0))
        shuffle_indices = tf.stack([shuffle_indices for _ in range(generated_images.shape[-1])], axis=-1)
        # for _ in range(2):
        #     shuffle_indices = tf.expand_dims(shuffle_indices, axis=-1)
        #choice_images = tf.expand_dims(choice_images, axis=1)
        stacked = tf.stack([choice_images, shuffle_indices], axis=1)

        def apply_gather(x):
            indices = x[1,0,0,:,0]
            return tf.gather(x, tf.cast(indices, tf.int32), axis=-2)

        stacked = tf.ensure_shape(stacked, shape=(32, 2, 28, 28, 2, 1))
        gathered = tf.map_fn(apply_gather, stacked)
        gathered = gathered[:, 0]
        choice_images = tf.stack(gathered, axis=0)

        #choice_images = tf.gather(choice_images, shuffle_indices, axis=0)
        #choice_images = tf.squeeze(choice_images, axis=1)

        # TODO: this is basically a repeat of the above, make it a function?  (it's basically zip for tf)
        choice_labels = tf.cast(choice_labels, dtype=tf.float32)
        shuffle_indices = tf.cast(shuffle_indices, dtype=tf.float32)
        # expanded_shuffle_indices = tf.identity(choice_images) #tf.zeros(shape=choice_images.shape)
        # expanded_shuffle_indices = tf.map_fn(lambda i: tf.fill(dims=expanded_shuffle_indices.shape[1:], value=i[0]), shuffle_indices)
        #for i in generated_images.shape[1:-2]:
        #    shuffle_indices = tf.stack([shuffle_indices for _ in range(i)],
        #                               axis=1)  # tf.tile(shuffle_indices, tf.expand_dims(i, axis=0))
        #shuffle_indices = tf.stack([shuffle_indices for _ in range(generated_images.shape[-1])], axis=-1)
        shuffle_indices = shuffle_indices[:, 0, 0, :, 0]
        # for _ in range(2):
        #     shuffle_indices = tf.expand_dims(shuffle_indices, axis=-1)
        # choice_images = tf.expand_dims(choice_images, axis=1)
        stacked = tf.stack([choice_labels, shuffle_indices], axis=1)

        def apply_gather(x):
            indices = x[1, :]
            return tf.gather(x, tf.cast(indices, tf.int32), axis=-1)

        stacked = tf.ensure_shape(stacked, shape=(32, 2, 2))
        gathered = tf.map_fn(apply_gather, stacked)
        gathered = gathered[:, 0]
        choice_labels = tf.stack(gathered, axis=0)

        #choice_labels = tf.gather(choice_labels, shuffle_indices, axis=0)
        return choice_images, choice_labels, tf.squeeze(generator_inputs, axis=-1), generated_images, target_images, shuffle_indices # images, labels, fake latents, fake images, true images, shuffle_indices

    def train(self, epochs, data_train, data_test, verbose=False, pretraining_steps = 0):
        self.data_train = data_train
        self.data_test = data_test
        steps_per_epoch = self.num_data // self.data_train.batch_size
        discriminator_updates = 2  # updates per discriminator update
        report_epoch_progress(None, None, self.generator, self.data_test)

        # manually set callbacks since we're doing a custom fit
        for callback in self.callbacks:
            callback.set_model(self.gan)

        for step in range(pretraining_steps):
            choice_images, choice_labels, _, _, _, _ = self.generate_choices()
            d_loss = self.discriminator.train_on_batch(choice_images, choice_labels)
            if verbose == True:
                print('PRETRAINING step ', step, '/', pretraining_steps, 'd_loss: ', d_loss)

        # custom training loop
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                choice_images, choice_labels, fake_latents, fake_images, true_images, shuffle_indices = self.generate_choices() # TODO: may be more efficient to split into real and fake generation and merging, since each is only used sometimes
                if step % discriminator_updates == 0:
                    d_loss = self.discriminator.train_on_batch(choice_images, choice_labels)
                # halt training on the discriminator
                self.discriminator.trainable = False
                # train generator to try to fool the discriminator
                g_loss = self.gan.train_on_batch([fake_latents, tf.squeeze(true_images, axis=-1), shuffle_indices], [true_images, np.random.uniform(.9, 1.0, size=(len(true_images),))])
                # and allow training on the discriminator to continue
                self.discriminator.trainable = True
                if verbose == True:
                    print('step ', step, '/', steps_per_epoch, 'd_loss: ', d_loss, 'g_loss: ', g_loss)
            # manually call callbacks since we're doing a custom fit
            logs = {'loss': g_loss}
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs, self.generator, self.data_test)
        # manually terminate callbacks since we're doing a custom fit
        for callback in self.callbacks:
            callback.on_train_end('_')


# ChoiceGAN paired with an autoencoder (acronym!)
class VEGAN():
    def __init__(self, g, d, e, num_choices=2):
        self.generator = g
        self.discriminator = d
        self.encoder = e
        self.num_data = 0
        self.batch_size=32
        self.num_choices = num_choices

        #gan_inputs = Input(shape=(100,))
        real_inputs = Input(shape=(28, 28, 1))
        encoded = self.encoder(real_inputs)
        gan_images = self.generator(encoded)
        shuffle_indices = Input(shape=(num_choices,)) # batch shape not included
        gan_choices = ToChoices()([real_inputs, gan_images, shuffle_indices])
        ToChoices().compute_output_shape([real_inputs.shape, gan_images.shape, shuffle_indices.shape])
        ToChoices().call([real_inputs, gan_images, shuffle_indices]) # TODO: remove this, for debugging
        gan_output = self.discriminator(gan_choices)

        self.encoder.summary()
        self.generator.summary()
        self.discriminator.summary()

        self.gan = Model(inputs=[real_inputs, shuffle_indices], outputs=[gan_images, gan_output])
        plot_model(self.gan, to_file='gan_model.png')

    def compile(self):
        d_optimizer = Adam(
            lr=.00001, beta_1=.5)  # lower lr necessary to keep discriminator from getting too much better than generator
        gan_optimizer = Adam(lr=.0001, beta_1=.5)

        self.discriminator.trainable = True
        self.discriminator.compile(optimizer=d_optimizer, loss='binary_crossentropy')
        self.discriminator.trainable = False
        loss = ['binary_crossentropy', 'mse']
        loss_weights = [100, 1]
        self.gan.compile(optimizer=gan_optimizer, loss=loss, loss_weights=loss_weights)
        self.discriminator.trainable = True

        # only the progress callback gets used right now, as I'm not sure how to make this work with a custom training loop
        progress_callback = LambdaCallback(on_epoch_end=report_epoch_progress)
        checkpoint_callback = ModelCheckpoint('./model-checkpoint.ckpt')
        tensorboard_callback = TensorBoard(log_dir='../logs/tensorboard-logs', write_images=True)
        self.callbacks = [progress_callback]
        self.gan.summary()

    # generate a list of images, half real, half fake, with the correct labels to train the discriminator
    # TODO: cleanup and convert numpy stuff to tf
    # image dims are (batch, x, y, channels, image)
    # label dims should be (batch, image)
    def generate_choices(self):
        # TODO: in the current implementation, should only work if the number of choices is even, but that's probably ok
        current_batch = [self.data_train.__getitem__(0) for _ in range(self.num_choices//2)]
        # generate "fake" images
        generator_inputs = np.stack([b[0] for b in current_batch], axis=-1)
        generated_images = np.stack([
            self.generator.predict(np.squeeze(gen_input), batch_size=self.batch_size) for gen_input in
            np.split(generator_inputs, generator_inputs.shape[-1], axis=-1)
        ], axis=-1)
        target_images = np.stack([b[1] for b in current_batch], axis=-1)
        #target_images = [b[1] for b in current_batch]
        #target_images = np.array(target_images)
        choice_images = tf.concat((generated_images, target_images), axis=-2) # merge the lists of real and fake images
        choice_labels = tf.cast(tf.stack(
            (
                np.random.uniform(0.0, .1, size=(len(generated_images),),),
                np.random.uniform(.9, 1.0, size=(len(current_batch[0][1]),))
            ), axis=-1), tf.float32) # merge the lists of real and fake labels
        shuffle_indices = tf.stack([tf.random.shuffle(tf.range(self.num_choices)) for _ in range(self.batch_size)], axis=0)

        choice_images = tf.cast(choice_images, dtype=tf.float32)
        shuffle_indices = tf.cast(shuffle_indices, dtype=tf.float32)
        # expanded_shuffle_indices = tf.identity(choice_images) #tf.zeros(shape=choice_images.shape)
        # expanded_shuffle_indices = tf.map_fn(lambda i: tf.fill(dims=expanded_shuffle_indices.shape[1:], value=i[0]), shuffle_indices)
        for i in generated_images.shape[1:-2]:
            shuffle_indices = tf.stack([shuffle_indices for _ in range(i)],
                                       axis=1)  # tf.tile(shuffle_indices, tf.expand_dims(i, axis=0))
        shuffle_indices = tf.stack([shuffle_indices for _ in range(generated_images.shape[-1])], axis=-1)
        # for _ in range(2):
        #     shuffle_indices = tf.expand_dims(shuffle_indices, axis=-1)
        #choice_images = tf.expand_dims(choice_images, axis=1)
        stacked = tf.stack([choice_images, shuffle_indices], axis=1)

        def apply_gather(x):
            indices = x[1,0,0,:,0]
            return tf.gather(x, tf.cast(indices, tf.int32), axis=-2)

        stacked = tf.ensure_shape(stacked, shape=(32, 2, 28, 28, 2, 1))
        gathered = tf.map_fn(apply_gather, stacked)
        gathered = gathered[:, 0]
        choice_images = tf.stack(gathered, axis=0)

        #choice_images = tf.gather(choice_images, shuffle_indices, axis=0)
        #choice_images = tf.squeeze(choice_images, axis=1)

        # TODO: this is basically a repeat of the above, make it a function?  (it's basically zip for tf)
        choice_labels = tf.cast(choice_labels, dtype=tf.float32)
        shuffle_indices = tf.cast(shuffle_indices, dtype=tf.float32)
        # expanded_shuffle_indices = tf.identity(choice_images) #tf.zeros(shape=choice_images.shape)
        # expanded_shuffle_indices = tf.map_fn(lambda i: tf.fill(dims=expanded_shuffle_indices.shape[1:], value=i[0]), shuffle_indices)
        #for i in generated_images.shape[1:-2]:
        #    shuffle_indices = tf.stack([shuffle_indices for _ in range(i)],
        #                               axis=1)  # tf.tile(shuffle_indices, tf.expand_dims(i, axis=0))
        #shuffle_indices = tf.stack([shuffle_indices for _ in range(generated_images.shape[-1])], axis=-1)
        shuffle_indices = shuffle_indices[:, 0, 0, :, 0]
        # for _ in range(2):
        #     shuffle_indices = tf.expand_dims(shuffle_indices, axis=-1)
        # choice_images = tf.expand_dims(choice_images, axis=1)
        stacked = tf.stack([choice_labels, shuffle_indices], axis=1)

        def apply_gather(x):
            indices = x[1, :]
            return tf.gather(x, tf.cast(indices, tf.int32), axis=-1)

        stacked = tf.ensure_shape(stacked, shape=(32, 2, 2))
        gathered = tf.map_fn(apply_gather, stacked)
        gathered = gathered[:, 0]
        choice_labels = tf.stack(gathered, axis=0)

        #choice_labels = tf.gather(choice_labels, shuffle_indices, axis=0)
        return choice_images, choice_labels, tf.squeeze(generator_inputs, axis=-1), generated_images, target_images, shuffle_indices # images, labels, fake latents, fake images, true images, shuffle_indices

    def train(self, epochs, data_train, data_test, verbose=False, pretraining_steps = 0):
        self.data_train = data_train
        self.data_test = data_test
        steps_per_epoch = self.num_data // self.data_train.batch_size
        discriminator_updates = 2  # updates per discriminator update
        report_epoch_progress(None, None, self.generator, self.data_test)

        # manually set callbacks since we're doing a custom fit
        for callback in self.callbacks:
            callback.set_model(self.gan)

        for step in range(pretraining_steps):
            choice_images, choice_labels, _, _, _, _ = self.generate_choices()
            d_loss = self.discriminator.train_on_batch(choice_images, choice_labels)
            if verbose == True:
                print('PRETRAINING step ', step, '/', pretraining_steps, 'd_loss: ', d_loss)

        # custom training loop
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                choice_images, choice_labels, fake_latents, fake_images, true_images, shuffle_indices = self.generate_choices() # TODO: may be more efficient to split into real and fake generation and merging, since each is only used sometimes
                if step % discriminator_updates == 0:
                    d_loss = self.discriminator.train_on_batch(choice_images, choice_labels)
                # halt training on the discriminator
                self.discriminator.trainable = False
                # train generator to try to fool the discriminator
                g_loss = self.gan.train_on_batch([tf.squeeze(true_images, axis=-1), shuffle_indices], [true_images, np.random.uniform(.9, 1.0, size=(len(true_images),))])
                # and allow training on the discriminator to continue
                self.discriminator.trainable = True
                if verbose == True:
                    print('step ', step, '/', steps_per_epoch, 'd_loss: ', d_loss, 'g_loss: ', g_loss)
            # manually call callbacks since we're doing a custom fit
            logs = {'loss': g_loss}
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs, self.generator, self.data_test)
        # manually terminate callbacks since we're doing a custom fit
        for callback in self.callbacks:
            callback.on_train_end('_')


# VEGAN, but where the encoded representation is appended to the generator output for the discriminator to judge
# the hope is that this will encourage the GAN to learn the fuller distribution of the data and avoid mode collapse
class DistributionalVEGAN():
    def __init__(self, g, d, e, num_choices=2):
        self.generator = g
        self.discriminator = d
        self.encoder = e
        self.num_data = 0
        self.batch_size=32
        self.num_choices = num_choices

        #gan_inputs = Input(shape=(100,))
        real_inputs = Input(shape=(28, 28, 1))
        encoded = self.encoder(real_inputs)
        gan_images = self.generator(encoded)
        shuffle_indices = Input(shape=(num_choices,)) # batch shape not included
        encoded_inputs = Input(shape=(100,))
        generated_inputs = Input(shape=(28, 28, 1))
        real_concater, fake_concater = ConcatWithEncoded(), ConcatWithEncoded()
        concated_reals, concated_fakes = real_concater([real_inputs, encoded]), fake_concater([gan_images, encoded])
        c_reals, c_fakes = real_concater([real_inputs, encoded_inputs]), fake_concater([generated_inputs, encoded_inputs])
        choice_maker = ToChoices()
        gan_choices = choice_maker([concated_reals, concated_fakes, shuffle_indices])
        c_choices = choice_maker([c_reals, c_fakes, shuffle_indices])
        ToChoices().compute_output_shape([real_inputs.shape, gan_images.shape, shuffle_indices.shape])
        ToChoices().call([concated_reals, concated_fakes, shuffle_indices]) # TODO: remove this, for debugging
        ConcatWithEncoded().call([real_inputs, encoded])
        gan_output = self.discriminator(gan_choices)

        concated_output = self.discriminator(c_choices)
        self.concated_discriminator = Model(inputs=[real_inputs, generated_inputs, encoded_inputs, shuffle_indices], outputs=[concated_output])

        self.encoder.summary()
        self.generator.summary()
        self.discriminator.summary()

        self.gan = Model(inputs=[real_inputs, shuffle_indices], outputs=[gan_images, gan_output])
        plot_model(self.gan, to_file='gan_model.png')

    def compile(self):
        d_optimizer = Adam(
            lr=.00001, beta_1=.5)  # lower lr necessary to keep discriminator from getting too much better than generator
        gan_optimizer = Adam(lr=.0001, beta_1=.5)

        self.discriminator.trainable = True
        self.concated_discriminator.trainable = True
        self.discriminator.compile(optimizer=d_optimizer, loss='binary_crossentropy')
        self.discriminator.trainable = False
        self.concated_discriminator.trainable = False

        # don't need to worry about setting trainable because all layers not shared with discriminator are non-trainable
        self.concated_discriminator.compile(optimizer=d_optimizer, loss='binary_crossentropy')

        loss = ['binary_crossentropy', 'mse']
        loss_weights = [100, 1]
        self.gan.compile(optimizer=gan_optimizer, loss=loss, loss_weights=loss_weights)
        self.discriminator.trainable = True
        self.concated_discriminator.trainable = True

        # only the progress callback gets used right now, as I'm not sure how to make this work with a custom training loop
        progress_callback = LambdaCallback(on_epoch_end=report_epoch_progress)
        checkpoint_callback = ModelCheckpoint('./model-checkpoint.ckpt')
        tensorboard_callback = TensorBoard(log_dir='../logs/tensorboard-logs', write_images=True)
        self.callbacks = [progress_callback]
        self.gan.summary()

    # generate a list of images, half real, half fake, with the correct labels to train the discriminator
    # TODO: cleanup and convert numpy stuff to tf
    # image dims are (batch, x, y, channels, image)
    # label dims should be (batch, image)
    def generate_choices(self):
        # TODO: in the current implementation, should only work if the number of choices is even, but that's probably ok
        current_batch = [self.data_train.__getitem__(0) for _ in range(self.num_choices//2)]
        # generate "fake" images
        generator_inputs = np.stack([b[0] for b in current_batch], axis=-1)
        generated_images = np.stack([
            self.generator.predict(np.squeeze(gen_input), batch_size=self.batch_size) for gen_input in
            np.split(generator_inputs, generator_inputs.shape[-1], axis=-1)
        ], axis=-1)
        target_images = np.stack([b[1] for b in current_batch], axis=-1)
        choice_images = tf.concat((generated_images, target_images), axis=-2) # merge the lists of real and fake images
        choice_labels = tf.cast(tf.stack(
            (
                np.random.uniform(0.0, .1, size=(len(generated_images),),),
                np.random.uniform(.9, 1.0, size=(len(current_batch[0][1]),))
            ), axis=-1), tf.float32) # merge the lists of real and fake labels
        shuffle_indices = tf.stack([tf.random.shuffle(tf.range(self.num_choices)) for _ in range(self.batch_size)], axis=0)

        choice_images = tf.cast(choice_images, dtype=tf.float32)
        shuffle_indices = tf.cast(shuffle_indices, dtype=tf.float32)
        for i in generated_images.shape[1:-2]:
            shuffle_indices = tf.stack([shuffle_indices for _ in range(i)],
                                       axis=1)  # tf.tile(shuffle_indices, tf.expand_dims(i, axis=0))
        shuffle_indices = tf.stack([shuffle_indices for _ in range(generated_images.shape[-1])], axis=-1)
        shuffle_indices = tf.cast(shuffle_indices, dtype=tf.float32)
        shuffle_indices = shuffle_indices[:, 0, 0, :, 0]

        return choice_images, choice_labels, tf.squeeze(generator_inputs, axis=-1), generated_images, target_images, shuffle_indices # images, labels, fake latents, fake images, true images, shuffle_indices

    def train(self, epochs, data_train, data_test, verbose=False, pretraining_steps = 0):
        self.data_train = data_train
        self.data_test = data_test
        steps_per_epoch = self.num_data // self.data_train.batch_size
        discriminator_updates = 2  # updates per discriminator update
        report_epoch_progress(None, None, self.generator, self.data_test)

        # manually set callbacks since we're doing a custom fit
        for callback in self.callbacks:
            callback.set_model(self.gan)

        for step in range(pretraining_steps):
            choice_images, choice_labels, _, _, _, _ = self.generate_choices()
            d_loss = self.discriminator.train_on_batch(choice_images, choice_labels)
            if verbose == True:
                print('PRETRAINING step ', step, '/', pretraining_steps, 'd_loss: ', d_loss)

        # custom training loop
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                choice_images, choice_labels, fake_latents, fake_images, true_images, shuffle_indices = self.generate_choices() # TODO: may be more efficient to split into real and fake generation and merging, since each is only used sometimes
                if step % discriminator_updates == 0:
                    # real_inputs, generated_inputs, encoded_inputs, shuffle_indices
                    real_ins = tf.squeeze(true_images, axis=-1)
                    encoded_ins = self.encoder.predict(real_ins)
                    generated_ins = self.generator.predict(encoded_ins)
                    d_loss = self.concated_discriminator.train_on_batch([real_ins, generated_ins, encoded_ins, shuffle_indices], choice_labels)
                # halt training on the discriminator
                self.discriminator.trainable = False
                self.concated_discriminator.trainable = False
                # train generator to try to fool the discriminator
                g_loss = self.gan.train_on_batch([tf.squeeze(true_images, axis=-1), shuffle_indices], [true_images, np.random.uniform(.9, 1.0, size=(len(true_images),))])
                # and allow training on the discriminator to continue
                self.discriminator.trainable = True
                self.concated_discriminator.trainable = True
                if verbose == True:
                    print('step ', step, '/', steps_per_epoch, 'd_loss: ', d_loss, 'g_loss: ', g_loss)
            # manually call callbacks since we're doing a custom fit
            logs = {'loss': g_loss}
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs, self.generator, self.data_test)
        # manually terminate callbacks since we're doing a custom fit
        for callback in self.callbacks:
            callback.on_train_end('_')