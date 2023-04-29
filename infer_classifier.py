from matplotlib import pyplot as plt
from PIL import Image
from time import time
import numpy as np
import argparse
import yaml
import os

from torchvision import transforms
import torch

from models.resnet import ResNet, BasicBlock

# load config
with open('./configs/config.yaml', 'r', encoding='utf-8') as f:
    yaml_info = yaml.load(f.read(), Loader=yaml.FullLoader)
classes = yaml_info['classes'].split()
n_classes = yaml_info['n_classes']
IMAGE_SIZE = yaml_info['image_size']
MEAN = yaml_info['mean']
STD = yaml_info['std']


def get_transform():
    """
    Returns:a composed transformation that converts an input image into a tensor
    and resizes it to a specified size

    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE]),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def preprocess_input(img):
    img_array = np.array(img)
    img_expand = np.expand_dims(img_array, 0)
    img = img_expand.transpose(1, 2, 0)
    img_transform = get_transform()(img)

    return img_transform.unsqueeze(0)


def infer(image_path, ckpt_path, device, visualize=False):
    # build model and load checkpoints
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=n_classes).to(device)
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    # read image and pre-process
    img = Image.open(image_path).convert('L')
    img_preprocess = preprocess_input(img)
    img_ = img_preprocess.to(device)
    # print(f"The shape of the image is: {img_.shape}")

    # forward
    with torch.no_grad():
        start_time = time()
        output = model(img_)
        end_time = time()

    print(f"Inference time: {1000. * (end_time - start_time):.2f}ms")

    # postprocess and get output
    _, out = torch.max(output, dim=1)
    label = classes[out.item()]
    print(f"Result: {label}")

    # visualization of result
    if visualize:
        plt.figure(figsize=(8, 8))
        plt.title(f"Result:{label}", fontsize=30)
        plt.imshow(img, 'gray')
        plt.show()


if __name__ == '__main__':
    # Define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, default='pretrained/resnet18-sam.pth',
                        help='checkpoints path for inference')
    parser.add_argument('--image-path', type=str, default='imgs/pneumonia_img2.png',
                        help='image path for inference')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='turn on flag to use GPU')
    parser.add_argument('--visualize', type=bool, default=True,
                        help='Enabling visualization. (default: True)')
    args = parser.parse_args()

    # Check input arguments
    if not os.path.exists(args.ckpt_path):
        print(f'Cannot find the checkpoints: {args.ckpt_path}')
        exit()
    if not os.path.exists(args.image_path):
        print(f'Cannot find the input image: {args.image_path}')
        exit()
    device = 'cuda' if args.use_gpu else 'cpu'

    # Run to inference
    infer(args.image_path, args.ckpt_path, device, args.visualize)
