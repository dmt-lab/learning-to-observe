from keras.applications import VGG16
from keras.layers import Conv2D, BatchNormalization, SpatialDropout2D, Activation, Concatenate, Input
from keras.models import Model
from models import building_blocks as bb

vgg16_branches = ('block1_pool', 'block2_pool', 'block3_pool', 'block4_pool')

def add_decoder_head(backbone_in, backbone_out, feats=256, n_classes=3, freeze_first=28, freeze_to_layer=None, n_upsample_levels=2, final_activation='softmax'):
    x = bb.upconv_bn_lrelu(backbone_out, feats, name='ucr0')
    mult = 1
    for l in range(n_upsample_levels):
        mult *= 2
        x = bb.upconv_bn_lrelu(x, int(feats/mult), name=f'ucr{l+1}')
    x = SpatialDropout2D(0.50)(x)
    head = Conv2D(n_classes, (1,1), name='head', padding='same', activation=final_activation)(x)
    model = Model(backbone_in, head)
    print(model.summary())
      
    if freeze_first is None and freeze_to_layer is not None:
        for layer in model.layers:
            print(layer.name)
            if not layer.name == freeze_to_layer:
                layer.trainable = False
            else:
                layer.trainable = False
                break

    elif freeze_first is not None:
        for layer in model.layers[:freeze_first]:
            layer.trainable = False
            print(f'Layer {layer.name} frozen')
    model.summary()
    return model


def to_multiscale_nin(backbone, input_shape, branch_names):
    '''
    Adapts VGG16 from a keras_applications fork into a multiscale VGG16.
    See paper: https://arxiv.org/pdf/1803.11395.pdf
    '''
    # Store mutliscale branch roots
    net_outputs = []
    for layer in backbone.layers:
        if layer.name in branch_names:
            net_outputs.append(layer.output)

    # Pass branches through 
    branch_outputs = []
    for branch, stride in zip(net_outputs, [4, 2, 1, 1]):
        x = bb.conv_bn_relu(branch, ch=128, ksize=3, stride=stride, activation='relu')
        x = bb.conv_bn_relu(x, ch=128, ksize=1, stride=1, activation='relu')
        x = bb.conv_bn_relu(x, ch=3, ksize=1, stride=1, activation='relu')
        branch_outputs.append(x)

    # Replace dense layers with conv2d 4x dilated
    x = Conv2D(512, (1,1), dilation_rate=(4,4), padding='same', activation=None)(backbone.output)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.1)(x)
    x = Conv2D(512, (1,1), dilation_rate=(4,4), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.1)(x)
    y = Conv2D(3, (1,1), name='output', activation='relu', padding='same')(x)

    # Merge all branches
    x = Concatenate(axis=-1)([y, *branch_outputs])

    multiscale_model = Model(backbone.inputs, x, name='MultiscaleVgg16')
    multiscale_model.summary()
    return multiscale_model


def vgg_to_multiscale(backbone, input_shape):
    model = to_multiscale_nin(backbone, input_shape, vgg16_branches)
    model = add_decoder_head(model.input, model.output,
        freeze_first=None,
        freeze_to_layer=None,
        final_activation='relu', n_upsample_levels=2,
        )
    return to_AET(model, input_shape)


def to_AET(backbone, input_shape):
    ori_img = Input(shape=input_shape)
    proc_img = Input(shape=input_shape)
    x1 = backbone(ori_img)
    x2 = backbone(proc_img)
    x = Concatenate(axis=-1)([x1, x2])
    x =  Conv2D(1, (1,1), name='output', activation='linear', padding='same')(x)
    return Model([ori_img, proc_img], x)


def build_aet(input_shape=(224,224,3)):
    vgg16 = VGG16(input_shape=input_shape, include_top=False, weights='imagenet', pooling=None, DILATED=True)
    return vgg_to_multiscale(vgg16, input_shape=input_shape)


if __name__ == "__main__":
    build_aet()