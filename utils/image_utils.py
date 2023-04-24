import numpy as np
import torch

from utils.classifier_dataset import PneumoniaDataset


# Function to decode tensor to numpy array
def decode_to_numpy(x):
    """
    This function takes a tensor as input and returns a numpy array after performing some operations on it.
    :param x: Input tensor
    :return: Numpy array after performing operations
    """
    x = x.cpu().detach().numpy()
    x_array = (x + 1) / 2
    x_array = (x_array - np.min(x_array)) / (np.max(x_array) - np.min(x_array))
    x_array = x_array * 255.

    return x, x_array


def getStat(train_data):
    """
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    """
    n_channels = 1

    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=False, num_workers=8,
        pin_memory=True)

    # initialize
    mean = torch.zeros(n_channels)
    std = torch.zeros(n_channels)
    for X, _ in train_loader:
        for d in range(n_channels):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    train_dataset = PneumoniaDataset('../data/real', '../data/fake', mode='train', transform=None)

    print(getStat(train_dataset))
