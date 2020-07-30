import os
os.environ['TF_CUDNN_DETERMINISTIC']='1'
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.metrics import mae
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam

from configs import aet_config
from generators.aet_generator import CocoSequence, TfCocoDataset
from models.aet import build_aet
from utils import utils
from utils.callbacks import SGDRScheduler
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Prep logdir
utils.mkdir(aet_config.LOG_DIR)
# Prep training run dir
job_dir, weights_dir, log_dir = utils.make_training_dirs(aet_config.LOG_DIR)

print(log_dir)

coco_train_gen = TfCocoDataset(
    batch_size=aet_config.BATCH_SIZE, 
    epochs=aet_config.EPOCHS,
    train=True,
    data_dir=aet_config.COCO_PATH
    )
coco_val_gen = TfCocoDataset(
    batch_size=aet_config.BATCH_SIZE, 
    epochs=aet_config.EPOCHS,
    train=False,
    data_dir=aet_config.COCO_PATH
    )


# Build model
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_aet(input_shape=aet_config.INPUT_SHAPE)
    optimizer = Adam(0.0001)

    if len(sys.argv) > 1 and sys.argv[1]:
        model.load_weights(sys.argv[1])

    model.compile(
        optimizer,
        loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
        metrics=[mae]
    )

steps = tf.data.experimental.cardinality(coco_train_gen).numpy()//aet_config.EPOCHS
val_steps = tf.data.experimental.cardinality(coco_val_gen).numpy()//aet_config.EPOCHS

# Prepare callbacks
tb_cback = TensorBoard(log_dir='logs')
model_ckpt = ModelCheckpoint(
    filepath=f'./{weights_dir}/best_aet_model.hdf5', 
    monitor='val_mean_squared_error', 
    save_best_only=True
    )
sgdr_cback = SGDRScheduler(
    min_lr=aet_config.SGDR_MIN_LR,
    max_lr=aet_config.SGDR_MAX_LR,
    steps_per_epoch=steps,
    lr_decay=aet_config.SGDR_DECAY,
    cycle_length=aet_config.SGDR_CYCLE_LENGTH
    )
callbacks = [tb_cback, sgdr_cback, model_ckpt]

# Train
model.fit(
    coco_train_gen,
    steps_per_epoch=steps,
    epochs=aet_config.EPOCHS,
    callbacks=callbacks,
    validation_data=coco_val_gen,
    validation_steps=val_steps,
    use_multiprocessing=False,
    workers=1,
    max_queue_size=64
)
