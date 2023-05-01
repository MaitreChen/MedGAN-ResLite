import yaml

import torchvision.transforms as transforms

# load config
with open('./configs/config.yaml', 'r', encoding='utf-8') as f:
    yaml_info = yaml.load(f.read(), Loader=yaml.FullLoader)
# classes = yaml_info['classes'].split()
n_classes = yaml_info['n_classes']
IMAGE_SIZE = yaml_info['image_size']
MEAN = yaml_info['mean']
STD = yaml_info['std']


def get_transform():
    """
    Returns:
        A composed transformation that converts an input image into a tensor
        and resizes it to a specified size. The transformation also normalizes
        the image using the specified mean and standard deviation.
    """
    return transforms.Compose([
        transforms.ToTensor(),  # convert image to tensor
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),  # resize image
        transforms.Normalize(mean=MEAN, std=STD)  # normalize image
    ])
