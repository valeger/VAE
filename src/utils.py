from typing import Dict, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_shape: Tuple):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        self.base_size = (128, output_shape[1] // 8, output_shape[2] // 8)
        self.fc = nn.Linear(latent_dim, np.prod(self.base_size))
        self.deconvs = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_shape[0], 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.fc(z)
        out = out.view(out.shape[0], *self.base_size)
        out = self.deconvs(out)
        return out


class ConvEncoder(nn.Module):
    def __init__(self, input_shape: Tuple, latent_dim: int):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.convs = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
        )
        conv_out_dim = input_shape[1] // 8 * input_shape[2] // 8 * 256
        self.fc = nn.Linear(conv_out_dim, 2 * latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.convs(x)
        out = out.view(out.shape[0], -1)
        mu, log_std = self.fc(out).chunk(2, dim=1)
        return mu, log_std


def train(model: nn.Module, 
          train_loader: DataLoader, 
          optimizer: optim.Adam, 
          epoch: int,
          device: torch.device,
          grad_clip=None) -> Dict:         
    model.train()

    losses = OrderedDict()
    for x in train_loader:
        x = x.to(device)
        out = model.loss(x)
        optimizer.zero_grad()
        out['loss'].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k, v in out.items():
            if k not in losses:
                losses[k] = []
            losses[k].append(v.item())
    return losses


def eval_loss(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    total_losses = OrderedDict()
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            out = model.loss(x)
            for k, v in out.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
    return total_losses


def train_epochs(model: nn.Module,
                 train_loader: DataLoader, 
                 test_loader: DataLoader, 
                 params: Dict) -> Tuple[Dict]:
    epochs, lr, device = params['epochs'], params['lr'], params['device']
    grad_clip = params.get('grad_clip', None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = OrderedDict(), OrderedDict()
    for epoch in range(epochs):
        model.train()
        train_loss = train(model, train_loader, optimizer, epoch, device, grad_clip)
        test_loss = eval_loss(model, test_loader, device)

        for k in train_loss.keys():
            if k not in train_losses:
                train_losses[k] = []
                test_losses[k] = []
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])
            print(f'Epoch {epoch}: train {k}: {np.mean(train_loss[k]):.4f}\n')
            print(f'Epoch {epoch}: test {k}: {test_loss[k]:.4f}\n')
        print('--------------------------------------------------')
    return train_losses, test_losses
