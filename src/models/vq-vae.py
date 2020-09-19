from typing import Dict, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from utils import *

class StackLayerNorm(nn.Module):
  def __init__(self, n_filters):
    super().__init__()
    self.h_layer_norm = LayerNorm(n_filters)
    self.v_layer_norm = LayerNorm(n_filters)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    vx, hx = x.chunk(2, dim=1)
    vx, hx = self.v_layer_norm(vx), self.h_layer_norm(hx)
    return torch.cat((vx, hx), dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )
      
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type: str, *args, conditional_size=None, **kwargs):
        assert mask_type == 'A' or mask_type == 'B'
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)

        if conditional_size is not None:
            self.cond_op = nn.Linear(conditional_size, self.out_channels)

    def forward(self, input: torch.Tensor, cond=None) -> torch.Tensor:
        out = F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        if cond is not None:
            cond = self.cond_op(cond)
            out = out + cond.view(cond.shape[0], self.out_channels, 1, 1)
        return out

    def create_mask(self, mask_type: str):
        k = self.kernel_size[0]
        self.mask[:, :, :k // 2] = 1
        self.mask[:, :, k // 2, :k // 2] = 1
        if mask_type == 'B':
            self.mask[:, :, k // 2, k // 2] = 1



class Quantize(nn.Module):

    def __init__(self, size, code_dim):
        super().__init__()
        self.embedding = nn.Embedding(size, code_dim)
        self.embedding.weight.data.uniform_(-1./size,1./size)

        self.code_dim = code_dim
        self.size = size

    def forward(self, z):
        b, c, h, w = z.shape
        weight = self.embedding.weight

        flat_inputs = z.permute(0, 2, 3, 1).contiguous().view(-1, self.code_dim)
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * torch.mm(flat_inputs, weight.t()) \
                    + (weight.t() ** 2).sum(dim=0, keepdim=True)
        encoding_indices = torch.max(-distances, dim=1)[1]
        encoding_indices = encoding_indices.view(b, h, w)
        quantized = self.embedding(encoding_indices).permute(0, 3, 1, 2).contiguous()

        return quantized, (quantized - z).detach() + z, encoding_indices


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()

class VectorQuantizedVAE(nn.Module):
    def __init__(self, code_dim, code_size):
        super().__init__()
        self.code_size = code_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        self.codebook = Quantize(code_size, code_dim)

        self.decoder = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode_code(self, x):
        with torch.no_grad():
            x = 2 * x - 1
            z = self.encoder(x)
            indices = self.codebook(z)[2]
            return indices

    def decode_code(self, latents):
        with torch.no_grad():
            latents = self.codebook.embedding(latents).permute(0, 3, 1, 2).contiguous()
            return self.decoder(latents).permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5

    def forward(self, x):
        z = self.encoder(x)
        e, e_st, _ = self.codebook(z)
        x_tilde = self.decoder(e_st)

        diff1 = torch.mean((z - e.detach()) ** 2)
        diff2 = torch.mean((e - z.detach()) ** 2)
        return x_tilde, diff1 + diff2

    def loss(self, x):
        x = 2 * x - 1
        x_tilde, diff = self(x)
        recon_loss = F.mse_loss(x_tilde, x)
        loss = recon_loss + diff
        return OrderedDict(loss=loss, recon_loss=recon_loss, reg_loss=diff)

class GatedConv2d(nn.Module):
  def __init__(self, mask_type, in_channels, out_channels, k=7, padding=3):
    super().__init__()

    self.vertical = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=k,
                              padding=padding, bias=False)
    self.horizontal = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(1, k),
                                padding=(0, padding), bias=False)
    self.vtoh = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1, 
                            bias=False)
    self.htoh = nn.Conv2d(out_channels, out_channels, kernel_size=1, 
                            bias=False)


    self.register_buffer('vmask', self.vertical.weight.data.clone())
    self.register_buffer('hmask', self.horizontal.weight.data.clone())

    self.vmask.fill_(1)
    self.hmask.fill_(1)

    # zero the bottom half rows of the vmask
    # No need for special color condition masking here since we get to see everything
    self.vmask[:, :, k // 2 + 1:, :] = 0

    # zero the right half of the hmask
    self.hmask[:, :, :, k // 2 + 1:] = 0
    if mask_type == 'A':
      self.hmask[:, :, :, k // 2] = 0
  
  def down_shift(self, x):
    x = x[:, :, :-1, :]
    pad = nn.ZeroPad2d((0, 0, 1, 0))
    return pad(x)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    vx, hx = x.chunk(2, dim=1)

    self.vertical.weight.data *= self.vmask
    self.horizontal.weight.data *= self.hmask

    vx = self.vertical(vx)
    hx_new = self.horizontal(hx)
    # Allow horizontal stack to see information from vertical stack
    hx_new = hx_new + self.vtoh(self.down_shift(vx))

    # Gates
    vx_1, vx_2 = vx.chunk(2, dim=1)
    vx = torch.tanh(vx_1) * torch.sigmoid(vx_2)

    hx_1, hx_2 = hx_new.chunk(2, dim=1)
    hx_new = torch.tanh(hx_1) * torch.sigmoid(hx_2)
    hx_new = self.htoh(hx_new)
    hx = hx + hx_new

    return torch.cat((vx, hx), dim=1)

# GatedPixelCNN using horizontal and vertical stacks to fix blind-spot
class GatedPixelCNN(nn.Module):
  def __init__(self, input_shape, code_size, dim=256, n_layers=7):
    super().__init__()
    self.n_channels = input_shape[0]
    self.input_shape = input_shape
    self.code_size = code_size

    self.embedding = nn.Embedding(code_size, dim)
    self.in_conv = MaskConv2d('A', dim, dim, 7, padding=3)
    model = []
    for _ in range(n_layers - 2):
      model.extend([nn.ReLU(), StackLayerNorm(dim), GatedConv2d('B', dim, dim, 7, padding=3)])
    model.extend([nn.ReLU(), StackLayerNorm(dim)])
    self.out_conv = MaskConv2d('B', dim, code_size, 7, padding=3)
    self.net = nn.Sequential(*model)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.embedding(x).permute(0, 3, 1, 2).contiguous()
    out = self.in_conv(out)
    out = self.net(torch.cat((out, out), dim=1)).chunk(2, dim=1)[1]
    out = self.out_conv(out)
    return out
  
  def loss(self, x: torch.Tensor) -> Dict:
    return OrderedDict(loss=F.cross_entropy(self(x), x.long()))

  def sample(self, n: int) -> torch.Tensor:
    samples = torch.zeros(n, *self.input_shape).long()
    with torch.no_grad():
      for r in range(self.input_shape[0]):
        for c in range(self.input_shape[1]):
            logits = self(samples)[:, :, r, c]
            probs = F.softmax(logits, dim=1)
            samples[:, r, c] = torch.multinomial(probs, 1).squeeze(-1)
    return samples


def train_all(train_data: np.ndarray, 
              test_data: np.ndarray,
              params: Dict) -> Tuple[np.ndarray]:    
    """
    Input:
      train_data: (n_train, 32, 32, 3) uint8 numpy array
      test_data: (n_test, 32, 32, 3) uint8 numpy array

    Returns
      a (# of training iterations,) numpy array of VQ-VAE train losess per minibatch
      a (# of epochs + 1,) numpy array of VQ-VAE train losses evaluated once at initialization and after each epoch
      a (# of training iterations,) numpy array of PixelCNN prior train losess per every minibatch
      a (# of epochs + 1,) numpy array of PixelCNN prior train losses evaluated once at initialization and after each epoch
      a (100, 32, 32, 3) numpy array of 100 samples if images
      a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
    """
    device = torch.device('cuda') if params['gpu'] else torch.device('cpu')
    params['device'] = device
    
    train_data = (np.transpose(train_data, (0, 3, 1, 2)) / 255).astype('float32')
    test_data = (np.transpose(test_data, (0, 3, 1, 2)) / 255).astype('float32')

    code_dim, code_size = 256, 128
    vqvae = VectorQuantizedVAE(code_dim, code_size).to(device)
    train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=128)
    train_losses, test_losses = train_epochs(vqvae, train_loader, test_loader,
                                             params)
    vqvae_train_losses, vqvae_test_losses = train_losses['loss'], test_losses['loss']

    def create_prior_dataset(data_loader):
        prior_data = []
        with torch.no_grad():
            for x in data_loader:
                x = x.to(device)
                z = vqvae.encode_code(x)
                prior_data.append(z.long())
        return torch.cat(prior_data, dim=0)

    prior = GatedPixelCNN(code_size=code_size, input_shape=(8, 8), dim=128, n_layers=15).to(device)
    prior_train_data, prior_test_data = create_prior_dataset(train_loader), create_prior_dataset(test_loader)
    prior_train_loader = data.DataLoader(prior_train_data, batch_size=128, shuffle=True)
    prior_test_loader = data.DataLoader(prior_test_data, batch_size=128)
    prior_train_losses, prior_test_losses = train_epochs(prior, prior_train_loader, prior_test_loader,
                                                         dict(epochs=10, lr=1e-3, grad_clip=1, device=device))
    prior_train_losses, prior_test_losses = prior_train_losses['loss'], prior_test_losses['loss']

    samples = prior.sample(100).long()
    samples = vqvae.decode_code(samples) * 255

    x = next(iter(test_loader))[:50].to(device)
    with torch.no_grad():
        z = vqvae.encode_code(x)
        x_recon = vqvae.decode_code(z)
    x = x.cpu().permute(0, 2, 3, 1).numpy()
    reconstructions = np.stack((x, x_recon), axis=1).reshape((-1, 32, 32, 3)) * 255

    return vqvae_train_losses, vqvae_test_losses, prior_train_losses, prior_test_losses, samples, reconstructions