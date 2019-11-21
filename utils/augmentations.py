import cv2
import numpy as np


def flip_augmentation(image, mask, chance=0.5):
    if np.random.rand() > chance:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if np.random.rand() > chance:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return image, mask


def zoom(image, mask, rng=(1.1, 1.5), chance=0.5):
    if np.random.rand() > chance:
        # Random scale and center
        scale_factor = np.random.random() * (max(rng) - min(rng)) + min(rng)
        new_dims = [int(dim / scale_factor) for dim in image.shape[:2]]
        random_row = np.random.randint( 0, (image.shape[0] - new_dims[0]) + 1 )
        random_col = np.random.randint( 0, (image.shape[1] - new_dims[1]) + 1 )
        # Perform crop
        image_crop = image[random_col:random_col+new_dims[0], random_row:random_row+new_dims[1], :]
        mask_crop = mask[random_col:random_col+new_dims[0], random_row:random_row+new_dims[1]]
        # Rescale to original resolution
        image_crop = cv2.resize(image_crop, image.shape[:2])
        mask_crop = cv2.resize(mask_crop, mask.shape[:2])
        return image_crop, mask_crop
    else:
        return image, mask
