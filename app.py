from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from submodels import *
from data import load_metadata, DataProvider
from gan import *

import random
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


def test_gan(gan_type, g_input, g_output, d_input, d_output, epochs=3, verbose=False):
    # we have to reinstantiate the generator and discriminator or the weights won't be reset
    generator = Model(g_input, g_output, name='generator')
    discriminator = Model(d_input, d_output, name='discriminator')
    gan_model = gan_type(generator, discriminator)
    gan_model.num_data = len(metadata)
    gan_model.compile()
    data_train = DataProvider(metadata_train)
    # build a dictionary mapping between name strings and ids
    data_train.class_to_id = dict((n, i) for i, n in enumerate(classes))
    data_train.id_to_class = dict((i, n) for i, n in enumerate(classes))
    data_test = DataProvider(metadata_test)
    data_test.class_to_id = dict((n, i) for i, n in enumerate(classes))
    data_test.id_to_class = dict((i, n) for i, n in enumerate(classes))
    gan_model.train(epochs, data_train, data_test, verbose=verbose)


random.seed()

# load data
metadata = load_metadata()
random.shuffle(metadata)
metadata_train = metadata[len(metadata)//10:]
metadata_test = metadata[:len(metadata_train)]

# get classes
classes = set([datum[0] for datum in metadata])
num_classes = len(classes)

test_gan(ChoiceGAN, *choicegan_submodel(), epochs=100, verbose=True)