# IMPORT PACKAGES
from PIL import Image
import yaml
import os

from torchvision import transforms
from torch.utils import data

# load config
with open('./configs/config.yaml', 'r', encoding='utf-8') as f:
    yaml_info = yaml.load(f.read(), Loader=yaml.FullLoader)

# set image size, mean and standard deviation from config file
IMAGE_SIZE = yaml_info['image_size']
MEAN = yaml_info['mean']
STD = yaml_info['std']


# resize 64x64 images
def preprocess_input(img):
    target_width = 64
    if img.height != target_width:
        img = img.resize((target_width, target_width))
    return img


# Pre-processing for ResNet18
class ImageTransform():
    def __init__(self):
        # define image transformations
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

    def __call__(self, img):
        # apply image transformations
        return self.data_transform(img)


class PneumoniaDataset(data.Dataset):
    def __init__(self, root1, root2, mode='train', transform=None):
        """
        Args:
            root1 (string): real dataset root
            root2 (string): fake dataset root
            mode (string, optional): train/val/test (default:train)
            transform (callable, optional): start transform
        """
        self.root1 = root1
        self.root2 = root2
        self.mode = mode
        self.transform = transform
        self.img_list = []
        self.label_list = []

        self.train_path1 = os.path.join(self.root1, 'train')
        self.val_path1 = os.path.join(self.root1, 'val')
        self.test_path1 = os.path.join(self.root1, 'test')

        # self.train_path2 = os.path.join(self.root2, 'train')
        # self.val_path2 = os.path.join(self.root2, 'val')
        # self.test_path2 = os.path.join(self.root2, 'test')

        # label: 0:normal , 1:pneumonia
        dataPath1 = {'train': self.train_path1, 'val': self.val_path1, 'test': self.test_path1}
        # dataPath2 = {'train': self.train_path2, 'val': self.val_path2, 'test': self.test_path2}

        # class1: normal images and label
        # real images (normal)
        normal_path1 = os.path.join(dataPath1[self.mode], 'normal')

        # modify the ratio of the test images
        normal_img_list1 = os.listdir(normal_path1)
        normal_img_list1_ = [os.path.join(normal_path1, _) for _ in normal_img_list1]
        normal_label_list1_ = [0] * len(normal_img_list1)

        # fake images (normal)
        normal_path2 = os.path.join(self.root2, 'normal', self.mode)
        normal_img_list2 = os.listdir(normal_path2)

        normal_img_list2_ = [os.path.join(normal_path2, _) for _ in normal_img_list2]
        normal_label_list2_ = [0] * len(normal_img_list2)

        # real + fake (normal)
        normal_img_list_ = normal_img_list1_ + normal_img_list2_
        normal_label_list_ = normal_label_list1_ + normal_label_list2_

        # class2: abnormal images and label
        pneumonia_path = os.path.join(dataPath1[self.mode], 'pneumonia')
        pneumonia_img_list = os.listdir(pneumonia_path)
        pneumonia_img_list_ = [os.path.join(pneumonia_path, _) for _ in pneumonia_img_list]
        pneumonia_label_list_ = [1] * len(pneumonia_img_list)

        print(f"{mode}")
        print(f"Number of normal images: {len(normal_img_list_)}")
        print(f"Number of pneumonia images: {len(pneumonia_img_list)}")

        # concatenate normal and pneumonia image lists and label lists
        self.img_list = normal_img_list_ + pneumonia_img_list_
        self.label_list = normal_label_list_ + pneumonia_label_list_

    def __len__(self):
        # return length of dataset
        return len(self.img_list)

    def __getitem__(self, index):
        # get image path and label at given index
        img_path = self.img_list[index]
        label = self.label_list[index]
        # open image and preprocess it
        img = Image.open(img_path).convert('L')
        img = preprocess_input(img)

        if self.transform is not None:
            # apply additional transformations if specified
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label



