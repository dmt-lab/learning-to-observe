from tensorflow.keras.layers import SpatialDropout2D, Conv2D
from tensorflow.keras.models import Model
from utils import utils

def freeze_layers(model, layer_name=None):
    for layer in model.layers:
        if not layer.name == layer_name:
            layer.trainable = False
            print(f'set {layer.name} to frozen')
        else:
            layer.trainable = False
            print(f'set {layer.name} to frozen')
            break
    return model


def get_backbone_layers(weights_path, nin_name='model_1', 
        input_layer_name='input_1', output_layer_name='activation_14'):
    backbone = utils.load_model(weights_path, compile=False)
    print(backbone.summary())
    if nin_name is None:
        backbone_in = backbone.get_layer(input_layer_name).input
        backbone_out = backbone.get_layer(output_layer_name).output
    else:
        print(backbone.get_layer(nin_name).summary())
        backbone_in = backbone.get_layer(nin_name).get_layer(input_layer_name).input
        backbone_out = backbone.get_layer(nin_name).get_layer(output_layer_name).output
        print(backbone.get_layer(nin_name).summary())
    return backbone_in, backbone_out


def build_classifier(backbone_path, freeze_to_layer=None):
    backbone_in, backbone_out = get_backbone_layers(backbone_path, nin_name='model_1', output_layer_name='rl2_ucr2')
    x = SpatialDropout2D(0.75, name='final_dropout')(backbone_out)
    head = Conv2D(3, (1,1), name='head_final', padding='same', activation='softmax')(x)
    model = Model(backbone_in, head)
    if freeze_to_layer is not None:
        model = freeze_layers(model, freeze_to_layer)
    return model

