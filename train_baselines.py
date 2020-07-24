import tensorflow as tf
from generators.aet_generator import TfCocoDataset

model = tf.keras.applications.ResNet50(input_shape=(224,224,3), include_top=False)

i1 = tf.keras.layers.Input(shape=(224,224,3))
i2 = tf.keras.layers.Input(shape=(224,224,3))

x1 = model(i1)
x2 = model(i2)

c = tf.keras.layers.Concatenate()([x1, x2])
x =  tf.keras.layers.Conv2D(1, (1,1), name='output', activation='linear', padding='same')(c)

aet = tf.keras.Model([i1, i2], x)

aet.compile('Adam', 'mse')

gen = TfCocoDataset()

aet.fit(gen)