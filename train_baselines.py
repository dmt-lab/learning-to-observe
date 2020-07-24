import tensorflow as tf
from generators.aet_generator import TfCocoDataset


def upsampling_block(input_tensor, channels):
    x = tf.keras.layers.UpSampling2D()(input_tensor)
    x = tf.keras.layers.Conv2D(channels, (4,4), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(channels, (4,4), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


model = tf.keras.applications.ResNet50(input_shape=(224,224,3), include_top=False)

x = upsampling_block(model.output, 512)
x = upsampling_block(x, 256)
x = upsampling_block(x, 128)
x = upsampling_block(x, 64)
x = upsampling_block(x, 32)

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
aet.compile('Adam', 'mse')

gen = TfCocoDataset()

aet.fit(gen)