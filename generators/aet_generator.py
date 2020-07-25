'''Sample and apply brightness transformations to images from COCO'''

import cv2
import numpy as np 
import skimage.io as io
from random import shuffle
from pycocotools.coco import COCO
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model
from utils.utils import quick_composite, preprocess_input
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import time
import tensorflow as tf


def imread(fname, shape):
    return cv2.resize(io.imread(fname) / 255.0, shape[:2])


class CocoSequence(Sequence):
    def __init__(self, batch_size=32, image_shape=(224, 224, 3), data_dir='E:/_________COCO_________',
            data_type='train2017', n_samples=None):
        annotation_file = f'{data_dir}/annotations/instances_{data_type}.json'
        self.data_dir = data_dir
        self.data_type = data_type
        self.batch_size = batch_size
        self.image_shape = image_shape

        # Initialize COCO
        self.coco = COCO(annotation_file)
        if n_samples is not None:
            self.x = self.coco.getImgIds()[:n_samples]
        else:
            self.x = self.coco.getImgIds()
        shuffle(self.x)
        self.n_images = len(self.x)

    def load_image(self, image_id):
        if not isinstance(image_id, list):
            image_id = [image_id]
        coco_img = self.coco.loadImgs(image_id)[0]
        fname = f'{self.data_dir}/images/{self.data_type}/{coco_img["file_name"]}'
        np_img = imread(fname, self.image_shape)
        return np_img, coco_img

    def load_mask(self, coco_img):
        ann_ids = self.coco.getAnnIds(imgIds=coco_img['id'], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        if len(anns) < 1:
            return None
        mask = cv2.resize(
            self.coco.annToMask(np.random.choice(anns)),
            self.image_shape[:2]
            )
        while np.sum(mask) < 0.01:
            mask = cv2.resize(
                self.coco.annToMask(np.random.choice(anns)),
                self.image_shape[:2]
                )
        return mask.astype(np.float32)

    def _composite(self, image, mask, rng=(np.log2(0.1), np.log2(10))):
        if len(image.shape) < 3: 
            return None
        if len(mask.shape) < 3:
            mask = np.dstack([mask] * 3) 
        offset = np.random.rand()  * np.abs(rng[0] - rng[1]) + rng[0]
        comp = quick_composite(image, mask, offset)
        return comp, offset

    def _generate(self, indices):
        X, X2, Y = [], [], []
        for idx in indices:
            image, coco_img = self.load_image(idx)
            mask = self.load_mask(coco_img)
            if mask is None:
                continue
            composite = self._composite(image, mask)
            if composite is None:
                continue
            composite, offset = composite
            X.append(image)
            X2.append(composite)
            Y.append(mask*offset)
        return X, X2, Y

    def __len__(self):
        return int(np.ceil(self.n_images / self.batch_size))

    def __getitem__(self, idx):
        indices = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        X, X2, Y = self._generate(indices)
        X = preprocess_input(X)
        X2 = preprocess_input(X2)
        batch_X = [
            np.array(X).reshape(-1, *self.image_shape),
            np.array(X2).reshape(-1, *self.image_shape)
            ]
        batch_Y = np.array(Y)
        return batch_X, batch_Y.reshape(-1, *self.image_shape[:2], 1)

    def tf_getitem(self, idx):
        indices = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        X, X2, Y = self._generate(indices)
        if X and X2 and Y:
            X = tf.keras.applications.resnet.preprocess_input(X[0])
            X2 = tf.keras.applications.resnet.preprocess_input(X2[0])
            # X = preprocess_input(X[0])
            # X2 = preprocess_input(X2[0])
            # print(type(Y))
            batch_Y = np.array(Y[0]).reshape((224,224,1))
            return np.array(X), np.array(X2), batch_Y
        else:
            return np.zeros((224,224,3)), np.zeros((224,224,3)), np.zeros((224,224,1))

    def on_epoch_end(self):
        np.random.shuffle(self.x)


def mapfn(idx, g):
    f = tf.py_function(func=g.tf_getitem, inp=[idx], Tout=[tf.float32, tf.float32, tf.float32])
    # f.set_shape()
    return (
        tf.ensure_shape(tf.convert_to_tensor(f[0]), (224,224,3)), 
        tf.ensure_shape(tf.convert_to_tensor(f[1]), (224,224,3))
        ), tf.ensure_shape(tf.convert_to_tensor(f[2]), (224,224,1))

def TfCocoDataset(train=True, data_dir='E:/mscoco', epochs=10, batch_size=32):
    g = CocoSequence(batch_size=1, data_dir=data_dir, data_type='train2017' if train else 'val2017')
    indices = list(range(g.n_images))

    ds = tf.data.Dataset.from_tensor_slices((indices))
    if train:
        ds = ds.shuffle(len(indices))
    ds = ds.map(lambda x: mapfn(x, g), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True if train else False)
    ds = ds.repeat(epochs)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


class CocoOrderedSequence(CocoSequence):

    def __init__(self, batch_size=32, image_shape=(224,224,3), data_dir='E:/_________COCO_________', data_type='train2017', n_samples=None, rng=(-3,3)):
        super().__init__(batch_size=batch_size, image_shape=image_shape, data_dir=data_dir, data_type=data_type, n_samples=n_samples)
        self.rng = rng

    def _generate(self, indices):
        X, X2, Y = [], [], []
        for idx in indices:
            image, coco_img = self.load_image(idx)
            if image.shape[-1] != 3:
                image = np.dstack([image] * 3)
            mask = self.load_mask(coco_img)
            if mask is None:
                continue
            for offset in np.linspace(*self.rng, 33):
                composite = quick_composite(image, mask, 2**offset)
                if composite is None:
                    continue
                X.append(image)
                X2.append(composite)
                Y.append(mask*offset)
        return X, X2, Y


if __name__ == "__main__":
    ds = TfCocoDataset()

    for x in ds:
        s = time.time()
        print(x[0].shape)
        print(f'Elapsed: {time.time()-s}')