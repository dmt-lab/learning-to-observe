import numpy as np
from models.aet import build_aet
from generators.aet_generator import CocoSequence
from keras.optimizers import Adam
from keras.losses import mse
from keras.metrics import mse, mae
from keras.callbacks import TensorBoard, ModelCheckpoint
from utils.callbacks import SGDRScheduler
from utils import utils

from keras.utils import multi_gpu_model

from configs import aet_config

# Prep logdir
utils.mkdir(aet_config.LOG_DIR)
# Prep training run dir
job_dir, weights_dir, log_dir = utils.make_training_dirs(aet_config.LOG_DIR)

# Prepare data
coco_train_gen = CocoSequence(
    batch_size=aet_config.BATCH_SIZE, 
    image_shape=aet_config.INPUT_SHAPE, 
    data_type='train2017', 
    n_samples=None, 
    data_dir=aet_config.COCO_PATH
    )
coco_val_gen = CocoSequence(
    batch_size=aet_config.BATCH_SIZE, 
    image_shape=aet_config.INPUT_SHAPE, 
    data_type='val2017', 
    n_samples=None, 
    data_dir=aet_config.COCO_PATH
    )



# Build model
model = build_aet(input_shape=aet_config.INPUT_SHAPE)
# Multi GPU
model = multi_gpu_model(model)
optimizer = Adam(0.0001)
model.compile(
    optimizer,
    loss=mse,
    metrics=[mse, mae]
)

# Prepare callbacks
tb_cback = TensorBoard(log_dir=log_dir, batch_size=aet_config.BATCH_SIZE)
model_ckpt = ModelCheckpoint(
    filepath=f'./{weights_dir}/best_aet_model.hdf5', 
    monitor='val_mean_squared_error', 
    save_best_only=True
    )
sgdr_cback = SGDRScheduler(
    min_lr=aet_config.SGDR_MIN_LR,
    max_lr=aet_config.SGDR_MAX_LR,
    steps_per_epoch=np.ceil(len(coco_train_gen) / aet_config.BATCH_SIZE),
    lr_decay=aet_config.SGDR_DECAY,
    cycle_length=aet_config.SGDR_CYCLE_LENGTH
    )
callbacks = [tb_cback, sgdr_cback, model_ckpt]

# Train
model.fit_generator(
    coco_train_gen,
    steps_per_epoch=len(coco_train_gen),
    epochs=aet_config.EPOCHS,
    callbacks=callbacks,
    validation_data=coco_val_gen,
    validation_steps=len(coco_val_gen),
    use_multiprocessing=False,
    workers=1,
    max_queue_size=64
)
