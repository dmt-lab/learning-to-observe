from keras.layers import UpSampling2D, Conv2D, BatchNormalization, Activation

def conv_bn_relu(input_x, ch, ksize, stride=1, dilation=1, activation='relu', padding='same', name=None):
    x = Conv2D(ch, (ksize, ksize), strides=(stride, stride), dilation_rate=(dilation, dilation), activation=None, padding=padding, name=name)(input_x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def upconv_bn_lrelu(input_tensor, n_channels, name=None):
    x = UpSampling2D(name=f'us_{name}')(input_tensor)
    x = Conv2D(n_channels, (3,3), padding='same', name=f'cv1_{name}')(x)
    x = BatchNormalization(name=f'bn1_{name}')(x)
    x = Activation('relu', name=f'rl1_{name}')(x)
    x = Conv2D(n_channels, (1,1), padding='same',name=f'cv2_{name}')(x)
    x = BatchNormalization(name=f'bn2_{name}')(x)
    x = Activation('relu', name=f'rl2_{name}')(x)
    return x