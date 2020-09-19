import os
import importlib
import argparse

import numpy as np
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from data.load import load_data

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='Name of a model to train',
                        default='vq-vae')
    parser.add_argument('--epochs', type=int, help='Number of epochs',
                        default=20)
    parser.add_argument('--lr', type=float, help='Learning rate',
                        default=1e-3)
    parser.add_argument('--gpu', type=bool, help='Enabling CUDA',
                        default=True)
    parser.add_argument('--grad_clip', type=int, 
                        help='Max norm of the gradients', default=1)
    return parser.parse_args().__dict__

def save_plot(train_losses: np.ndarray, 
              test_losses: np.ndarray, 
              title: str) -> None:
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    try:
        os.makedirs('images')
    except:
        pass

    fname = os.path.join('images', 'train_plot.png')
    plt.savefig(fname, format='png')
    return None

def save_samples(samples: torch.Tensor, title: str, nrow=10) -> None:
    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')

    try:
        os.makedirs('images')
    except:
        pass

    fname = os.path.join('data', f'{title}.png')
    plt.savefig(fname, format='png')
    return None

if __name__ == '__main__':
    params = parse()

    try:
        model = importlib.import_module(f'models.{params["name"]}')
    except ImportError:
        print('This model does not exist')

    train_data, test_data = load_data()
    if params['name'] == 'vq-vae':
        train_losses, test_losses, prior_train_losses, prior_test_losses, samples, reconstructions = \
            model.train_all(train_data, test_data, params)
    else:
        train_losses, test_losses, samples, reconstructions, interps = model.train_all(train_data, test_data, params)

    save_plot(train_losses, test_losses, f'{params["name"].upper()} Train Plot',)
    if params['name'] == 'vq-vae':
        save_plot(prior_train_losses, prior_test_losses, f'VQ-VAE PixelCNN Prior Train Plot',)
    save_samples(samples, 'Samples')
    save_samples(reconstructions, 'Reconstructions')
