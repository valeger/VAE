from typing import Tuple

import numpy as np
import torchvision

Arrays = Tuple[np.ndarray]

def load_data() -> Arrays:
    train_data = torchvision.datasets.CIFAR10('./data', train=True, 
        transform=torchvision.transforms.ToTensor(), download=True)
    test_data = torchvision.datasets.CIFAR10('./data', train=True, 
        transform=torchvision.transforms.ToTensor(), download=True)
        
    return train_data.data, test_data.data