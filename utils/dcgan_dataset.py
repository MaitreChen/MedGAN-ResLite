# IMPORT PACKAGES
from matplotlib import pyplot as plt
from PIL import Image
import os
import yaml


from torchvision import transforms
from torch.utils import data
import torchvision.utils

from utils.common_utils import print_separator

# load config
with open('./configs/config.yaml', 'r', encoding='utf-8') as f:
    yaml_info = yaml.load(f.read(), Loader=yaml.FullLoader)

IMAGE_SIZE = yaml_info['dcgan_image_size']
MEAN = yaml_info['mean']
STD = yaml_info['std']


def load_image_paths_for_class(data_dir, class_name):
    """
    Load file paths of image data for a specified class from the ChestXRay-2017 dataset.

    Args:
    - data_dir (str): Directory containing the dataset.
    - class_name (str): Name of the class to load ('normal' or 'pneumonia').

    Returns:
    - img_list (list): List of file paths for images of the specified class.
    """
    img_list = []

    # Get the directory path for the specified class
    class_dir = os.path.join(data_dir, class_name)
    # class_dir = os.path.join(data_dir)

    # Loop through train and val folders
    for data_split in ['train', 'val']:
        split_dir = os.path.join(data_dir, data_split, class_name)

        # Check if the split directory exists
        if not os.path.exists(split_dir):
            continue

        # Loop through image files
        for img_file in os.listdir(split_dir):
            img_path = os.path.join(split_dir, img_file)
            img_list.append(img_path)

    return img_list


# Pre-processing for DCGAN
class ImageTransformDCGAN():
    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), InterpolationMode.BICUBIC),
            transforms.Normalize((0.5,), (0.5,)),  # if you want to train normal, open it
            # transforms.Normalize(mean=MEAN, std=STD),
            # transforms.RandomVerticalFlip(p=0.5),


            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomRotation(15),
            # transforms.RandomApply([transforms.RandomErasing(p=0.2)], p=0.2),
            # transforms.ColorJitter(0.5, 0.5),
            # transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.2),
        ])

    def __call__(self, img):
        return self.data_transform(img)


class DCGANDataset(data.Dataset):

    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    # Return the number of images
    def __len__(self):
        return len(self.file_list)

    # Return preprocessed data in tensor format
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        # Convert image to grayscale
        img_gray = img.convert('L')

        # Apply transformations
        img_transformed = self.transform(img_gray)

        return img_transformed


def get_dataloader_dcgan(data_dir, class_name, batch_size, num_workers):
    print_separator()
    print('==> Getting dataloader..')

    # Create file list
    train_img_list = load_image_paths_for_class(data_dir, class_name)

    # Create Dataset
    train_dataset = DCGANDataset(
        file_list=train_img_list, transform=ImageTransformDCGAN())
    print("dataset: ", len(train_dataset))

    # Create dataloader
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataloader


def show_images(images):
    # Create a grid of images
    grid = torchvision.utils.make_grid(images, padding=5, normalize=True)

    # Convert tensor to numpy array
    grid = grid.numpy().transpose((1, 2, 0))

    # Display the grid
    fig = plt.figure(figsize=(5, 5))
    fig.suptitle("Pulmonary Images", fontsize=17)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # # Example usage:
    DATA_DIR = '../data/chest_xray2017_prepared'
    CLASS_NAME = 'normal'
    BATCH_SIZE = 1
    NUM_WORKERS = 16
    # pneumonia_img_list = load_image_paths_for_class(data_dir, 'pneumonia')

    train_dataloader = get_dataloader_dcgan(DATA_DIR, CLASS_NAME, BATCH_SIZE, NUM_WORKERS)
    # batch_iterator = iter(train_dataloader)
    # real_images = next(batch_iterator)

    # torchvision.utils.save_image(real_images, 'real.png')
    # show_images(real_images)
