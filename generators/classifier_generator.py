import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import utils
from utils import augmentations as aug
from keras.utils import to_categorical, Sequence


def rand_exp(rng=(1, 3), power=3, cnt=1):
    x = np.random.rand(cnt)
    y = power**x
    z_b = y - 1
    z_b = z_b / (power - 1)
    z = z_b * (rng[1]-rng[0]) + rng[0]
    return z[0]


def rand_loguniform(start=None, end=None):
    assert start is not None and end is not None, 'Start and end values must be specified' 
    n = np.random.rand() * np.abs(start-end) + start
    assert n >= start and n <= end
    return n


class Composite:
    def __init__(self, image_path, mask_path, low_threshold, high_threshold, image_shape=(224,224,3), augment=True):
        self.image_path = image_path
        self.mask_path = mask_path
        self.low_threshold = np.log2(low_threshold)
        self.high_threshold = np.log2(high_threshold)
        self.image_shape = image_shape
        self.image = utils.load_image(self.image_path, image_shape)
        self.mask = utils.clean_mask(utils.load_image(self.mask_path, image_shape))
        self.augment = augment

    def make_mask(self, offset):
        new_mask = np.zeros(self.image_shape)
        if offset < self.low_threshold:
            new_mask[:,:,0] = self.mask
            new_mask[:,:,2] = 1.0 - self.mask
        elif offset > self.high_threshold:
            new_mask[:,:,1] = self.mask
            new_mask[:,:,2] = 1.0 - self.mask
        else:
            new_mask[:,:,2] = 1.0
        return new_mask

    def _generate(self, offset):
        comp = utils.quick_composite(self.image, self.mask, np.power(2,offset))
        mask = self.make_mask(offset)
        if self.augment:
            comp, mask = aug.flip_augmentation(comp, mask, chance=0.5)
            comp, mask = aug.zoom(comp, mask, chance=0.5)
        comp = utils.preprocess_input(comp)
        return comp, mask

    def generate_subthreshold(self):
        offset = rand_loguniform(self.low_threshold, self.high_threshold)
        return self._generate(offset)

    def generate_positive(self):
        offset = rand_exp(rng=(self.high_threshold, np.log2(10)))
        return self._generate(offset)

    def generate_negative(self):
        offset = rand_exp(rng=(np.log2(0.1), self.low_threshold))
        return self._generate(offset)



class CompositeSequence(Sequence):
    def __init__(self, thresholds_csv, batch_size, shuffle_pre_training=False, shuffle_on_epoch=True, image_shape=(224,224,3), train_val='train'):
        self.thresholds_csv = thresholds_csv
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.shuffle_pre_training = shuffle_pre_training
        self.shuffle_on_epoch = shuffle_on_epoch
        self.composites = self._prepare_composites()
        if train_val == 'train':
            self.composites = self.composites[:int(0.9*len(self.composites))]
            print(f'Number of training composites: {len(self.composites)}')
        else:
            self.composites = self.composites[int(0.9*len(self.composites)):]
            print(f'Number of validation composites: {len(self.composites)}')

        [print(comp.image_path for comp in self.composites)]

    def _prepare_composites(self):
        composites = []
        for idx, dataset_row in pd.read_csv(self.thresholds_csv).iterrows():
            image_path = dataset_row['image_path']
            name, extension = image_path.split('.')
            mask_path = f'{name}m.{extension}'
            composites.append(
                Composite(
                    image_path, 
                    mask_path, 
                    dataset_row['negative_threshold'], 
                    dataset_row['positive_threshold'],
                    image_shape=self.image_shape
                )
            )
        return composites

    def _generate_batch(self, comp):
        Xn, Yn = comp.generate_negative()
        Xp, Yp = comp.generate_positive()
        Xs, Ys = comp.generate_subthreshold()
        return [Xn, Xp, Xs], [Yn, Yp, Ys]

    def __len__(self):
        return int(np.ceil(len(self.composites) / float(self.batch_size)))#

    def __getitem__(self, idx):
        batch_size = int(np.floor(self.batch_size / 3))
        batch_x = self.composites[idx * batch_size:(idx + 1) * batch_size]
        X = []
        y = []
        for item in batch_x:
            X_list, Y_list = self._generate_batch(item)
            X.extend(X_list)
            y.extend(Y_list)
        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.shuffle_on_epoch:
            np.random.shuffle(self.composites)


def test_generator():
    gen = CompositeSequence('./data/perceptual_thresholds.csv', 3)
    for batch in gen:
        X, Y = batch
        for x, y in zip(X,Y):
            _, ax = plt.subplots(1,4)
            ax[0].imshow(x)
            ax[1].imshow(y[:,:,0])
            ax[2].imshow(y[:,:,1])
            ax[3].imshow(y[:,:,2])
            plt.show()


