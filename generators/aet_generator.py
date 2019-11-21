'''Sample and apply brightness transformations to images from COCO'''

import cv2
import numpy as np 
import skimage.io as io
from random import shuffle
from pycocotools.coco import COCO
from keras.utils import Sequence
from utils.utils import quick_composite, preprocess_input


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

    def on_epoch_end(self):
        np.random.shuffle(self.x)