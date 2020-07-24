import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from keras.models import load_model
from generators.aet_generator import CocoOrderedSequence

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./data/pretrained_aet_model.hdf5')
    parser.add_argument('--output', default='./data/latest_aet_results.npy')
    parser.add_argument('--data_dir', default='E:/mscoco')

    return parser.parse_args()

def generate_validation_results(model_path='./data/pretrained_aet_model.hdf5', 
data_dir='E:/mscoco', rng=(-1,1), output_path='./data/latest_aet_results.npy'):
    gen = CocoOrderedSequence(data_dir=data_dir, batch_size=1, n_samples=None, data_type='val2017', rng=rng)
    model = load_model(model_path, compile=False)
    mses = np.empty((len(gen), 33))
    for idx, (x, y) in enumerate(gen):
        p = model.predict_on_batch(x)
        if p.shape[0] > 0:
            error = np.mean(np.abs(y - p), axis=(1,2,3))
            mses[idx,:] = error
    np.save(output_path, mses, allow_pickle=True)
    plt.plot(np.linspace(*rng, 33), np.mean(mses, axis=0))
    plt.plot(np.linspace(*rng, 33), np.max(mses, axis=0))
    plt.show()


def plot_results(result_path='./data/aet_eval_11.npy', rng=(-1,1)):
    x = np.load(result_path)
    with plt.style.context(['science', 'ieee']):
        print(x.shape)
        print(np.mean(x, axis=0))
        print(np.std(x, axis=0))
        plt.figure(figsize=(4, 1))
        plt.errorbar(
            np.linspace(*rng, 33), 
            np.mean(x, axis=0), 
            np.std(x, axis=0), 
            elinewidth=0.5, 
            capsize=1, 
            fmt='o-', 
            capthick=0.5, 
            linewidth=0.5, 
            markersize=0.75
        )
        plt.xlabel('Exposure shift (stops)')
        plt.ylabel('MSE')
        plt.savefig(f'./data/thesis_images/{os.path.basename(result_path).split(".")[0]}.png')
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    generate_validation_results(
        model_path=args.model,
        data_dir=args.data_dir, 
        rng=(-1, 1), 
        output_path=args.output
        )
    plot_results(result_path=args.output, rng=(-1,1))