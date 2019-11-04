from tensorflow.keras.utils import Sequence
from skimage import io
import numpy as np
import csv
import re
import random
from skimage.transform import downscale_local_mean


data_root = 'D:/HDD Data/emnist/emnist_byclass_train/'


def load_metadata():
    metadata = []
    with open(data_root + 'metadata.csv', 'r', newline='') as metadata_file:
        reader = csv.reader(metadata_file, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for r, row in enumerate(reader):
            if len(row) == 0:
                continue
            if r > 50000:
                break
            metadatum = row[1]
            metadata.append(metadatum)
    return metadata


class DataProvider(Sequence):
    metadata = None
    batch_size = 32

    def __init__(self, metadata):
        self.metadata = metadata
        self.class_to_id = None
        self.id_to_class = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        x, y = self.get_batch(self.batch_size, self.metadata)
        if x is None or y is None:
            raise ValueError("input x or y is none")
        return x, y

    # note to self for the future: optimize the batches to use np arrays from the getgo?
    def get_batch(self, batch_size, metadata):
        img_x, img_y = 28, 28
        batch_y = np.zeros((batch_size, img_x, img_y), dtype='float32')
        for i in range(batch_size):
            datum_index = random.randint(0, len(metadata) - 1)
            img_scaled = self.get_image(metadata, datum_index % len(metadata))
            metadatum = metadata[datum_index % len(metadata)]

            # put it in a tensor after downscaling it and padding it
            img_downscaled = img_scaled
            # normalize channel values
            batch_y[i] = img_downscaled
            batch_y[i] /= 255.0
        return np.random.uniform(size=(batch_size, 100)), np.expand_dims(batch_y, axis=3)

    def get_image(self, metadata, datum_index):
        metadatum = metadata[datum_index]
        return io.imread(data_root + 'images/' + metadatum[0] + '.png', as_gray=True)
