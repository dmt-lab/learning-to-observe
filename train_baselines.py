import os
os.environ['TF_CUDNN_DETERMINISTIC']='1'
import tensorflow as tf
import tensorflow as tf
from generators.aet_generator import TfCocoDataset
from configs import aet_config
import numpy as np
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def upsampling_block(input_tensor, channels, skip=None):
    if skip is not None:
        input_tensor = tf.keras.layers.Conv2D(channels, (1,1), padding='same')(input_tensor)
        skip = tf.keras.layers.Conv2D(channels, (1,1), padding='same')(skip)
        input_tensor = tf.keras.layers.Add()([input_tensor, skip])
    x = tf.keras.layers.UpSampling2D()(input_tensor)
    x = tf.keras.layers.Conv2D(channels, (4,4), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(channels, (4,4), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


skips = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = tf.keras.applications.ResNet50(input_shape=(224,224,3), include_top=False)

    x = upsampling_block(model.output, 512, model.get_layer(skips[-1]).output)
    x = upsampling_block(x, 256, model.get_layer(skips[-2]).output)
    x = upsampling_block(x, 128, model.get_layer(skips[-3]).output)
    x = upsampling_block(x, 64, model.get_layer(skips[-4]).output)
    x = upsampling_block(x, 32, model.get_layer(skips[-5]).output)

    model = tf.keras.Model(model.input, x)
    model.summary()

    i1 = tf.keras.layers.Input(shape=(224,224,3))
    i2 = tf.keras.layers.Input(shape=(224,224,3))

    x1 = model(i1)
    x2 = model(i2)

    c = tf.keras.layers.Concatenate()([x1, x2])
    x =  tf.keras.layers.Conv2D(1, (1,1), name='output', activation='linear', padding='same')(c)

    aet = tf.keras.Model([i1, i2], x)
    aet.summary()

    opt = tf.keras.optimizers.Adam(lr=0.001)
    aet.compile(opt, 'mse')

gen = TfCocoDataset(data_dir='/home/dolhasz/coco', batch_size=aet_config.BATCH_SIZE)
val_gen = TfCocoDataset(data_dir='/home/dolhasz/coco', batch_size=aet_config.BATCH_SIZE, train=False)

steps = tf.data.experimental.cardinality(gen).numpy()//aet_config.EPOCHS
val_steps = tf.data.experimental.cardinality(val_gen).numpy()//aet_config.EPOCHS

aet.fit(gen, steps_per_epoch=steps, validation_data=val_gen, validation_steps=val_steps)