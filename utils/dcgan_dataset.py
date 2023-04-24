# IMPORT PACKAGES
from PIL import Image
import os

from torchvision import transforms
from torch.utils import data


def make_data_path_list():
    """
    # Create file path lists of image data & annotation
    """

    # The file path to save the image data
    data_path = './data/real/'
    img_list = []

    # Train two classes separately
    # Class1: normal
    # train and val
    label_name = 'normal'
    train_normal_path_list = os.listdir(os.path.join(data_path, 'train', label_name))
    num_of_train_images = len(train_normal_path_list)
    for i in range(1, num_of_train_images):
        img_path = f"{data_path}train/{label_name}/img_" + str(i) + '.png'
        img_list.append(img_path)

    val_normal_path_list = os.listdir(os.path.join(data_path, 'val', label_name))
    num_of_val_images = len(val_normal_path_list)
    for i in range(1214, num_of_val_images):
        img_path = f"{data_path}val/{label_name}/img_" + str(i) + '.png'
        img_list.append(img_path)

    # # Class2: pneumonia
    # label_name = 'pneumonia'
    # train_abnormal_path_list = os.listdir(os.path.join(data_path, 'train', label_name))
    # num_of_train_images = len(train_abnormal_path_list)
    # for i in range(1, num_of_train_images):
    #     img_path = f"{data_path}train/{label_name}/img_" + str(i) + '.png'
    #     img_list.append(img_path)
    #
    # val_abnormal_path_list = os.listdir(os.path.join(data_path, 'val', label_name))
    # num_of_val_images = len(val_abnormal_path_list)
    # for i in range(3495, num_of_val_images):
    #     img_path = f"{data_path}val/{label_name}/img_" + str(i) + '.png'
    #     img_list.append(img_path)
    '''
    train_path_list = os.listdir(os.path.join(data_path, 'train'))
    num_of_train_images = len(train_path_list)
    
    val_path_list = os.listdir(os.path.join(data_path, 'val'))
    num_of_val_images = len(val_path_list)
    '''

    '''
    # for cDCGAN
    for p in train_path_list:
        img_path = f"{data_path}train/{p}"
        img_list.append(img_path)

    # val
    for p in val_path_list:
        img_path = f"{data_path}val/{p}"
        img_list.append(img_path)
    '''

    return img_list


# Pre-processing for DCGAN
class ImageTransformDCGAN():
    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),

            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomRotation(15),
            # transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        ])

    def __call__(self, img):
        return self.data_transform(img)


class GAN_Img_Dataset(data.Dataset):

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

        # preprocessing
        img_transformed = self.transform(img)

        return img_transformed


def get_dataloader_dcgan(batch_size=64, workers=8):
    # Create file list
    train_img_list = make_data_path_list()

    # Create Dataset
    train_dataset = GAN_Img_Dataset(
        file_list=train_img_list, transform=ImageTransformDCGAN())
    print("dataset: ", len(train_dataset))

    # Create dataloader
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    return train_dataloader
