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
IMG_H, IMG_W = 64, 128

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


def crop_imgs(imgs, boxs):
    crops = np.ndarray(shape=(len(imgs), IMG_H, IMG_W),
                       dtype=np.float32)
    for i in range(len(imgs)):
        # x = np.nanmin(np.nonzero(boxs[i][:, 0]))
        # y = np.nanmin(np.nonzero(boxs[i][:, 1]))
        # w = np.nanmax(boxs[i][:, 2])
        # h = np.nanmax(boxs[i][:, 3])
        x = int(round(boxs[i][0]))
        y = int(round(boxs[i][1]))
        x2 = int(round(boxs[i][2]))
        y2 = int(round(boxs[i][3]))
        img = imgs[i][y:y2, x:x2]
        img = ndimage.zoom(img, [IMG_H / (y2-y), IMG_W / (x2-x)])
        if crops[i].shape != img.shape: print(i, crops[i].shape, img.shape, x,y,x2,y2)
        crops[i, 0:img.shape[0], 0:img.shape[1]] = img

    return crops


def read_img(filepath):
    try:
        image_data = ndimage.imread(filepath).astype(float)
        image_data = image_data[:,:,0] * 0.2989 + image_data[:,:,1] * 0.5870 + image_data[:,:,2] * 0.114
        # image_data = np.mean(image_data, -1)
        image_data = (image_data - 255. / 2) / 255.
        hp = IMG_H / image_data.shape[0]
        wp = IMG_W / image_data.shape[1]
        image_data = ndimage.zoom(image_data, [hp, wp])
        return image_data, hp, wp
    except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')


def read_imgs(path):
    st = h5py.File(path + '/digitStruct.mat', 'r')
    name = st['/digitStruct/name']
    bbox = st['/digitStruct/bbox']

    dataset = np.ndarray(shape=(len(name), IMG_H, IMG_W),
                         dtype=np.float32)
    labset = np.ndarray(shape=(len(name), 5, 11),
                        dtype=np.float32)
    boxs = np.ndarray(shape=(len(name), 4),
                      dtype=np.float32)
    for i in range(len(name)):
        img_file = ''.join([chr(c[0]) for c in st[name[i][0]].value])
        img_file = os.path.join(path, img_file)
        # read labels
        label = st[bbox[i][0]]['label']

        def get_attr(attr):
            if len(attr) == 1:
                return [int(attr[0][0])]
            else:
                return [int(st[ attr[j][0] ][0][0]) for j in range(len(attr))]
        label_onehot = np.zeros([5, 11], dtype=np.float32)
        labels = get_attr(label)
        xs = get_attr(st[bbox[i][0]]['left'])
        ys = get_attr(st[bbox[i][0]]['top'])
        ws = get_attr(st[bbox[i][0]]['width'])
        hs = get_attr(st[bbox[i][0]]['height'])
        if len(labels) > 5:
            print('skip', i, labels)
            continue
        for j in range(min(len(labels), 5)):
            n = labels[j]
            if n == 10: n = 0
            label_onehot[j, n] = 1
            # boxs[i, j, 0] = xs[j]
            # boxs[i, j, 1] = ys[j]
            # boxs[i, j, 2] = ws[j]
            # boxs[i, j, 3] = hs[j]
        # read image
        image_data, hp, wp = read_img(img_file)
        if image_data is None:
            continue
        boxs[i, 0] = np.min(xs) * wp
        boxs[i, 1] = np.min(ys) * hp
        boxs[i, 2] = np.max(np.add(xs,ws)) * wp
        boxs[i, 3] = np.max(np.add(ys,hs)) * hp
        # print(boxs[i],np.min(xs) ,np.max(xs+ws)  ,wp)
        for j in range(len(label), 5):
            label_onehot[j, 10] = 1
        labset[i, :, :] = label_onehot
        dataset[i, :, :] = image_data
    return dataset, labset, boxs


def read_data_sets(train_dir, validation_size=500):
    TRAIN_IMAGES = 'train.tar.gz'
    TEST_IMAGES = 'test.tar.gz'

    local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                     SOURCE_URL + TRAIN_IMAGES)

    local_path = maybe_extract(local_file)[0]
    train_images, train_labels, train_boxs = read_imgs(local_path)

    local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                     SOURCE_URL + TEST_IMAGES)

    local_path = maybe_extract(local_file)[0]
    test_images, test_labels, test_boxs = read_imgs(local_path)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    validation_boxs = train_boxs[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]
    train_boxs = train_boxs[validation_size:]

    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'train_boxs':train_boxs,

        'validation_images': validation_images,
        'validation_labels': validation_labels,
        'validation_boxs': validation_boxs,

        'test_images': test_images,
        'test_labels': test_labels,
        'test_boxs': test_boxs
    }
