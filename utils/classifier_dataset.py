# IMPORT PACKAGES
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import yaml
import os
import cv2

from torchvision import transforms
from torch.utils import data

from utils.common_utils import print_separator

# load config
with open('./configs/config.yaml', 'r', encoding='utf-8') as f:
    yaml_info = yaml.load(f.read(), Loader=yaml.FullLoader)


# resize 64x64 images
def preprocess_input(img):
    target_width = 64
    if img.height != target_width:
        img = img.resize((target_width, target_width))
    return img


class PneumoniaOriginalDataset(data.Dataset):
    def __init__(self, root, mode='train', transform=None):
        """
        Args:
            root (string): Dataset root directory
            mode (string, optional): train/val/test (default:train)
            transform (callable, optional): Transformations to apply to the data
        """
        self.root = root
        self.mode = mode
        self.transform = transform
        self.img_list = []
        self.label_list = []

        self._load_data()

    def _load_data(self):
        data_path = os.path.join(self.root, self.mode)
        normal_path = os.path.join(data_path, 'normal')
        pneumonia_path = os.path.join(data_path, 'pneumonia')

        normal_img_list = os.listdir(normal_path)
        pneumonia_img_list = os.listdir(pneumonia_path)

        self.img_list.extend([os.path.join(normal_path, img) for img in normal_img_list])
        self.img_list.extend([os.path.join(pneumonia_path, img) for img in pneumonia_img_list])

        self.label_list.extend([0] * len(normal_img_list))  # Normal images have label 0
        self.label_list.extend([1] * len(pneumonia_img_list))  # Pneumonia images have label 1

        print(f"-------------{self.mode}-------------")
        print(f"Number of normal images: {len(normal_img_list)}")
        print(f"Number of pneumonia images: {len(pneumonia_img_list)}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]

        # Load image using cv2
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the image
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        # img = clahe.apply(img)

        # Convert image to PIL format
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


class PneumoniaBalancedDataset(data.Dataset):
    def __init__(self, real_root, fake_root, mode='train', transform=None):
        """
        Args:
            real_root (string): The root of the original data set
            fake_root (string): The generated fake image root directory
            mode (string, optional): train/val/test (default: train)
            transform (callable, optional): Apply to data augmentation
        """
        self.real_root = real_root
        self.fake_root = fake_root
        self.mode = mode
        self.transform = transform
        self.img_list = []
        self.label_list = []

        self.merge_fake1 = mode != 'test'  # Flag whether to merge fake images
        self.merge_fake2 = mode != 'val'

        self._load_data1()
        # self._load_data2()

    def _load_data1(self):
        real_data_path = os.path.join(self.real_root, self.mode).replace('\\', '/')
        fake_data_path = os.path.join(self.fake_root, self.mode).replace('\\', '/')

        real_normal_path = os.path.join(real_data_path, 'normal').replace('\\', '/')
        pneumonia_path = os.path.join(real_data_path, 'pneumonia').replace('\\', '/')

        fake_normal_path = os.path.join(fake_data_path, 'normal').replace('\\', '/')

        normal_img_list = os.listdir(real_normal_path)
        pneumonia_img_list = os.listdir(pneumonia_path)

        self.img_list.extend([os.path.join(real_normal_path, img).replace('\\', '/') for img in normal_img_list])
        self.img_list.extend([os.path.join(pneumonia_path, img).replace('\\', '/') for img in pneumonia_img_list])

        self.label_list.extend([0] * len(normal_img_list))  # Normal images have label 0
        self.label_list.extend([1] * len(pneumonia_img_list))  # Pneumonia images have label 1

        # Load fake normal images if needed
        if self.merge_fake1 and self.merge_fake2:
            fake_normal_img_list = os.listdir(fake_normal_path)

            self.img_list.extend(
                [os.path.join(fake_normal_path, img).replace('\\', '/') for img in fake_normal_img_list])
            self.label_list.extend([0] * len(fake_normal_img_list))  # Fake normal images have label 0

        print(f"---------------{self.mode}---------------")
        print(f"Number of real normal images: {len(normal_img_list)}")
        print(f"Number of real pneumonia images: {len(pneumonia_img_list)}")

        if self.merge_fake1 and self.merge_fake2:
            print(f"Number of fake normal images: {len(fake_normal_img_list)}")
            print(f"Number of total normal images: {len(normal_img_list) + len(fake_normal_img_list)}")
            print(f"Number of total pneumonia images: {len(pneumonia_img_list)}")

    def _load_data2(self):
        real_data_path = os.path.join(self.real_root, self.mode).replace('\\', '/')
        fake_data_path = os.path.join(self.fake_root, self.mode).replace('\\', '/')

        real_normal_path = os.path.join(real_data_path, 'normal').replace('\\', '/')
        real_pneumonia_path = os.path.join(real_data_path, 'pneumonia').replace('\\', '/')

        fake_normal_path = os.path.join(fake_data_path, 'normal').replace('\\', '/')
        fake_pneumonia_path = os.path.join(fake_data_path, 'pneumonia').replace('\\', '/')

        normal_img_list = os.listdir(real_normal_path)
        pneumonia_img_list = os.listdir(real_pneumonia_path)

        self.img_list.extend([os.path.join(real_normal_path, img).replace('\\', '/') for img in normal_img_list])
        self.img_list.extend([os.path.join(real_pneumonia_path, img).replace('\\', '/') for img in pneumonia_img_list])

        self.label_list.extend([0] * len(normal_img_list))  # Normal images have label 0
        self.label_list.extend([1] * len(pneumonia_img_list))  # Pneumonia images have label 1

        # Load fake normal images if needed
        if self.merge_fake1 and self.merge_fake2:
            fake_normal_img_list = os.listdir(fake_normal_path)
            fake_pneumonia_img_list = os.listdir(fake_pneumonia_path)

            self.img_list.extend(
                [os.path.join(fake_normal_path, img).replace('\\', '/') for img in fake_normal_img_list])
            self.label_list.extend([0] * len(fake_normal_img_list))  # Fake normal images have label 0

            self.img_list.extend(
                [os.path.join(fake_pneumonia_path, img).replace('\\', '/') for img in fake_pneumonia_img_list])
            self.label_list.extend([0] * len(fake_pneumonia_img_list))  # Fake normal images have label 0

        print(f"-------------{self.mode}-------------")
        print(f"Number of real normal images: {len(normal_img_list)}")
        print(f"Number of real pneumonia images: {len(pneumonia_img_list)}")

        if self.merge_fake1 and self.merge_fake2:
            print(f"Number of fake normal images: {len(fake_normal_img_list)}")
            print(f"Number of fake pneumonia images: {len(fake_pneumonia_img_list)}")
            print(f"Number of total normal images: {len(normal_img_list) + len(fake_normal_img_list)}")
            print(f"Number of total pneumonia images: {len(pneumonia_img_list) + len(fake_pneumonia_img_list)}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]

        # Load image using cv2
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the image
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        # img = clahe.apply(img)

        # Convert image to PIL format
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


def show_samples(dataset, num_samples=6):
    """
    Display specified number of image samples from the dataset.

    Parameters:
    dataset: Dataset object.
    num_samples: Number of image samples to display, default is 6.

    Returns:
    None.
    """

    # Set up subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Display first three images with label 0
    for i in range(3):
        # Get image and label
        image, label = dataset[i]

        # Convert PIL image to NumPy array
        image = transforms.ToPILImage()(image)

        # Display image and label in subplot
        axes[0, i].imshow(image, cmap='gray')
        axes[0, i].set_title('Label: {}'.format(label))
        axes[0, i].axis('off')

    # Display last three images with label 1
    for i in range(3):
        # Get image and label
        image, label = dataset[i + 1215]  # Add the number of samples with label 0 before

        # Convert PIL image to NumPy array
        image = transforms.ToPILImage()(image)

        # Display image and label in subplot
        axes[1, i].imshow(image, cmap='gray')
        axes[1, i].set_title('Label: {}'.format(label))
        axes[1, i].axis('off')

    # Adjust subplot spacing
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real_data_dir = os.path.join(project_root, "data", "real")
    fake_data_dir = os.path.join(project_root, "data", "fake2")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create an instance of PneumoniaDatasetOriginal with the defined transformation
    # original_dataset = PneumoniaOriginalDataset(root=real_data_dir, mode='train', transform=transform)
    balanced_dataset = PneumoniaBalancedDataset(real_root=real_data_dir, fake_root=fake_data_dir, mode='train',
                                                transform=transform)

    show_samples(balanced_dataset, num_samples=6)
