import os
import argparse
import numpy as np
from utils import utils
from utils.metrics import mean_dice_np
from generators.classifier_generator import CompositeSequence
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from configs import classifier_config

def parse_args():
    parser = argparse.ArgumentParser(description='Perform inference using trained models')
    parser.add_argument('--model', help='Path to trained threshold classifier model')
    parser.add_argument('--image', help='Path to single input image, directory or tree of directories')
    parser.add_argument('--dataset', help='''Run inference on training or validation dataset 
        - displays ground truth masks, as well as predictions. Do not use if already using --image.''',
        choices=['train', 'val']
        )
    return parser.parse_args()


def calculate_f1(y_true, y_pred):
    if y_true.ndim < 3:
        y_true = np.dstack([y_true] * 3)
    params = {
        'y_true' : utils.stretch(y_true), 
        'y_pred' : utils.stretch(y_pred), 
        'drop_last' : False, 
        'mean_per_class' : True, 
        'metric_type' : 'soft'
        }
    return mean_dice_np(**params)


def predict_single_image(model, image, image_size=(224,224,3)):
    preproc = utils.preprocess_input(image)
    prediction = utils.squeeze(model.predict(utils.stretch(preproc)))
    return prediction


def predict_folder(model, folder_path):
    image_list = utils.recursive_find_images(args.image, extensions=('.jpg', '.png', '.jpeg'))
    results = []
    for image_path in image_list:
        image = utils.load_image(image_path)
        pred = predict_single_image(model, image_path)
        results.append((image, pred))
    return results


def predict_dataset(model, dataset_name):
    generator = CompositeSequence(classifier_config.THRESHOLDS_CSV, batch_size=1, train_val=dataset_name)
    results = []
    for comp in generator.composites:
        print(os.path.basename(comp.image_path))
        image_results = [] 
        image = comp.image
        mask = comp.mask
        for offset in np.linspace(-1, 1, 21):
            composite = utils.quick_composite(image, mask, np.power(2,offset))
            prediction = predict_single_image(model, composite)
            image_results.append((
                composite, 
                mask if offset > comp.high_threshold or offset < comp.low_threshold else np.zeros_like(mask),
                prediction
                ))
        results.append(image_results)
    return results


def visualise_gt(results):
    n_composites = len(results)
    n_offsets = len(results[0])
    n_subimages = len(results[0][0])
    for comp in results:
        plt.figure(figsize=(20, 4))
        plt.margins(y=20)
        grid = gridspec.GridSpec(n_subimages+1, n_offsets, top=0.95, bottom=0.3)
        grid.update(wspace=0.1, hspace=0.1)  # set the spacing between axes.
        fscores = []
        for off_idx, offset in enumerate(comp):
            fscores.append(calculate_f1(offset[1], offset[2]))
            for si_idx, subimg in enumerate(offset):
                ax = plt.subplot(grid[si_idx, off_idx])
                ax.imshow(subimg, vmin=0.0, vmax=1.0)
                ax.set_xticks([])
                ax.set_yticks([])
        ax = plt.subplot(grid[3:, :])
        fscores = np.array(fscores)
        ax.plot(np.linspace(-1,1,21), fscores[:, 0], label='Class 0 F1 score')
        ax.plot(np.linspace(-1,1,21), fscores[:, 1], label='Class 1 F1 score')    
        plt.xlim([-1.05, 1.05])
        plt.ylim([0, 1])
        plt.ylabel('F1 score')
        plt.xlabel('Exposure offset ($stops$)')
        plt.legend(loc='lower right')
        plt.show()


def main(args):
    model, input_shape = utils.load_model_get_shape(args.model)
    print('Loaded model') 
    if args.dataset is not None:
        print('Predicting dataset')
        results = predict_dataset(model, args.dataset)
        visualise_gt(results)
    elif os.path.isdir(args.image):
        results = predict_folder(model, args.image)
    else: 
        image = utils.load_image(args.image)
        results = predict_single_image(model, image)


if __name__ == "__main__":
    print('what')
    args = parse_args()
    main(args)
