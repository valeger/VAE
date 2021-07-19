# VAE Models

This project is about different small Variational Autoencoder models trained on CIFAR10 dataset.

What exactly are these models?

* a standard VAE (convvae)
* [Variational Lossy Autoencoder](https://arxiv.org/abs/1611.02731) (vlae)
* [Vector Quantised-Variational AutoEncoder](https://arxiv.org/abs/1711.00937) (vq-vae)


## Run

1. Install all dependencies listed in requirements.txt. Note that the model has only been tested in the versions shown in the text file.
2. Choose an appropriate model 
    * `convvae` stands for classic Variational AutoEncoder
    * `vlae` stands for Variational Lossy AutoEncoder 
    * `vq-vae` stands for Vector Quantised-Variational AutoEncoder (default)

```bash
cd src && python3 main.py --name vlae
```
As far as the output, several plots will be saved in `images` directory (train plot + samples + reconstruction images)

Training stage takes about 3 hours (on GPU NVIDIA GeForce GTX 1080 Ti).