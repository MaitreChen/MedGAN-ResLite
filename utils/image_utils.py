import numpy as np
import cv2
import os

from torchvision import transforms
import torch

from utils.classifier_dataset import PneumoniaOriginalDataset
from utils.classifier_dataset import PneumoniaBalancedDataset


def resize_grayscale_images_in_directory(directory_path, target_size=(64, 64)):
    # check dir
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    # get all files in dir
    file_list = os.listdir(directory_path)

    # for loop files
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)

        # check image
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            try:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, target_size)
                cv2.imwrite(file_path, img_resized)

                print(f"Resized '{file_name}' to {target_size}")
            except Exception as e:
                print(f"Error processing '{file_name}': {e}")
        else:
            print(f"Skipping non-image file '{file_name}'")


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
    :param train_data: Dataset
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
    # Set definition
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize([224, 224]),/
    ])

    # train_dataset = PneumoniaOriginalDataset('../data/real', transform=transform)
    train_dataset = PneumoniaBalancedDataset('../data/real', '../data/fake', mode='train', transform=transform)

    print(getStat(train_dataset))
