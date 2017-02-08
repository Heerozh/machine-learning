from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
from scipy import ndimage


import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
import h5py

SOURCE_URL = 'http://ufldl.stanford.edu/housenumbers/'
IMG_H, IMG_W = 66, 150

def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(root+'/')
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    print(data_folders)
    return data_folders



def read_imgs(path):
    st = h5py.File(path + '/digitStruct.mat', 'r')
    name = st['/digitStruct/name']
    bbox = st['/digitStruct/bbox']

    dataset = np.ndarray(shape=(len(name), IMG_H, IMG_W),
                         dtype=np.float32)
    labset = np.ndarray(shape=(len(name), 5, 11),
                         dtype=np.float32)
    for i in range(len(name)):
        img_file = ''.join([chr(c[0]) for c in st[name[i][0]].value])
        img_file = os.path.join(path, img_file)

        label = st[bbox[i][0]]['label']
        if len(label) == 1:
            labels = [int(label[0][0])]
        else:
            labels = [int(st[ label[j][0] ][0][0]) for j in range(len(label))]
        label_onehot = np.zeros([5, 11], dtype=np.float32)
        if len(labels) > 5:
            print('skip', i, labels)
            continue
        for i in range(min(len(labels), 5)):
            label_onehot[i, labels[i]] = 1
        for i in range(len(label), 5):
            label_onehot[i, 10] = 1
        labset[i, :, :] = label_onehot
        try:
            image_data = ndimage.imread(img_file).astype(float)
            # image_data = (image_data - 255. / 2) / 255.
            image_data = np.mean(image_data, -1)
            h = image_data.shape[0]
            w = image_data.shape[1]
            image_data = ndimage.zoom(image_data, [IMG_H / h, IMG_W / w])
            dataset[i, :, :] = image_data
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    return dataset, labset


def read_data_sets(train_dir, validation_size=5000):
    TRAIN_IMAGES = 'train.tar.gz'
    TEST_IMAGES = 'test.tar.gz'

    local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                     SOURCE_URL + TRAIN_IMAGES)

    local_path = maybe_extract(local_file)[0]
    train_images, train_lables = read_imgs(local_path)

    local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                     SOURCE_URL + TEST_IMAGES)

    local_path = maybe_extract(local_file)[0]
    test_images, test_lables = read_imgs(local_path)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    return {
        'train_images': train_images,
        'train_lables': train_labels,

        'validation_images': validation_images,
        'validation_labels': validation_labels,
        'test_images': test_images,
        'test_lables': test_lables
    }
