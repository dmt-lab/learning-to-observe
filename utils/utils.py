import os
import datetime
import cv2

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


def load_image(path, target_size=None):
    if target_size is not None and len(target_size) > 2:
        target_size = (target_size[0], target_size[1])
    img = image.img_to_array(image.load_img(path, target_size=target_size)) / 255.0
    return img


def preprocess_input(x):
    if isinstance(x, list):
        return [img * 2 - 1.0 for img in x]
    else:
        return x * 2.0 - 1.0


def stretch(image):
    return np.expand_dims(image, axis=0)


def squeeze(image):
    return np.squeeze(image)


def clean_mask(mask):
    mask = np.mean(mask, axis=-1)
    mask = np.where(mask < 0.2, 0.0, 1.0)
    return mask
    

def calc_object_size(mask):
    nonzero = np.count_nonzero(mask)
    all_elements = len(mask.reshape(-1))
    return nonzero / all_elements


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def make_training_dirs(train_dir):
    job_dir = mkdir(os.path.join(train_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')))
    weights_dir = mkdir(os.path.join(job_dir, 'weights'))
    log_dir = mkdir(os.path.join(job_dir, 'logs'))
    return job_dir, weights_dir, log_dir


def load_model_get_shape(model_path):
    model = load_model(model_path, compile=False)
    input_shape = model.layers[0].input_shape[1:]
    return model, input_shape


def recursive_find_images(path, extensions=('.jpg', '.png')):
    results = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.lower().endswith(extensions):
                    results.append(os.path.join(root, name))
    return results


def quick_composite(image, mask, offset):
    if mask.ndim < 3:
        mask  = np.dstack([mask] *3)
    fg = cv2.cvtColor((image.copy() * 255).astype(np.uint8), cv2.COLOR_RGB2Lab)
    fg[:,:,0] = np.clip(fg[:,:,0] * offset, 0, 255)
    fg = cv2.cvtColor(fg, cv2.COLOR_Lab2RGB) / 255.
    comp = fg * mask + image * (1-mask)
    comp = np.clip(comp, 0.0, 1.0)
    return comp