"""Prepare mnist seq."""
import random
import numpy as np
import input_data


MNIST = input_data.read_data_sets("MNIST_data/", one_hot=True)

MNIST_IMG_SZ = 28
DIGITS = 0x0A

def gen_seq_img(images, labels, seqs=5):
    """generate one seqed img"""
    img = np.ndarray([MNIST_IMG_SZ, 0], dtype=np.float32)
    lab = np.zeros([seqs, DIGITS], dtype=np.float32)
    rndlen = random.randint(1, seqs)
    for i in range(0, seqs):
        if i > rndlen:
            char = np.zeros_like(images[0]).reshape(MNIST_IMG_SZ, MNIST_IMG_SZ)
        else:
            rndsel = random.randint(0, len(images)-1)
            char = images[rndsel].reshape(MNIST_IMG_SZ, MNIST_IMG_SZ)
            lab[i, np.argmax(labels[rndsel])] = 1
        img = np.hstack((img, char))
    return img, lab

def gen_datas(size, test=False, seqs=5):
    """generate dataset"""
    x_dat = np.ndarray([size, MNIST_IMG_SZ, MNIST_IMG_SZ * seqs, 1], dtype=np.float32)
    y_dat = np.ndarray([size, seqs, DIGITS], dtype=np.float32)
    for i in range(0, size):
        images = MNIST.test.images if test else MNIST.train.images
        labels = MNIST.test.labels if test else MNIST.train.labels
        seqimg, seqlab = gen_seq_img(images, labels, seqs)
        x_dat[i, :, :, :] = seqimg.reshape(MNIST_IMG_SZ, MNIST_IMG_SZ * seqs, 1)
        y_dat[i, :, :] = seqlab
    return x_dat, y_dat

