import numpy as np
from models.classifier import build_classifier
from generators.classifier_generator import CompositeSequence
from segmentation_models.losses import CategoricalFocalLoss
from utils.metrics import mean_iou
from utils import utils
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from utils.callbacks import SGDRScheduler

from configs import classifier_config

# Prep logdir
utils.mkdir(classifier_config.LOG_DIR)
# Prep training run dir
job_dir, weights_dir, log_dir = utils.make_training_dirs(classifier_config.LOG_DIR)


# Prepare data
train_gen = CompositeSequence(
    thresholds_csv=classifier_config.THRESHOLDS_CSV,
    batch_size=classifier_config.BATCH_SIZE,
    shuffle_pre_training=False,
    shuffle_on_epoch=True,
    image_shape=classifier_config.INPUT_SHAPE,
    train_val='train'
    )

val_gen = CompositeSequence(
    thresholds_csv=classifier_config.THRESHOLDS_CSV,
    batch_size=classifier_config.BATCH_SIZE,
    shuffle_pre_training=False,
    shuffle_on_epoch=False,
    image_shape=classifier_config.INPUT_SHAPE,
    train_val='val'
    )

print(f'Training generator with {len(train_gen)} batches')
print(f'Validation generator with {len(val_gen)} batches')
print((len(train_gen)+len(val_gen))*classifier_config.BATCH_SIZE)

# Build model
model = build_classifier(classifier_config.AET_PATH, freeze_to_layer='concatenate_1')
optimizer = Adam(0.0001)
model.compile(
    optimizer,
    loss=CategoricalFocalLoss(alpha=0.25, gamma=4),
    metrics=[mean_iou]
)

# Prepare callbacks
tb_cback = TensorBoard(log_dir=log_dir, batch_size=classifier_config.BATCH_SIZE)
model_ckpt = ModelCheckpoint(
    filepath=f'./{weights_dir}/best_classifier_model.hdf5', 
    monitor='val_mean_iou', 
    save_best_only=True
    )
sgdr_cback = SGDRScheduler(
    min_lr=classifier_config.SGDR_MIN_LR,
    max_lr=classifier_config.SGDR_MAX_LR,
    steps_per_epoch=np.ceil(len(train_gen) / classifier_config.BATCH_SIZE),
    lr_decay=classifier_config.SGDR_DECAY,
    cycle_length=classifier_config.SGDR_CYCLE_LENGTH
    )
early_stopping = EarlyStopping(monitor='val_loss', patience=400)
callbacks = [tb_cback, sgdr_cback, model_ckpt, early_stopping]

# Train
model.fit_generator(
    train_gen,
    steps_per_epoch=len(train_gen),
    epochs=classifier_config.EPOCHS,
    callbacks=callbacks,
    validation_data=val_gen,
    validation_steps=len(val_gen),
    use_multiprocessing=False,
    workers=1,
    max_queue_size=64
)