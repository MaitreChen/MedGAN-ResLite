from PIL import Image
from time import time
import numpy as np
import cv2 as cv
import argparse
import os

import torch

from models.resnet import ResNet, BasicBlock
from utils.common_utils import create_folder
from utils.get_data_info import get_transform, n_classes


def preprocess_input(img):
    """
    Preprocesses the input image by converting it to a tensor and applying
    a composed transformation to it.

    Args:
        img: The input image

    Returns:
        The preprocessed image
    """
    img_array = np.array(img)
    img_expand = np.expand_dims(img_array, 0)
    img = img_expand.transpose(1, 2, 0)
    img_transform = get_transform()(img)

    return img_transform.unsqueeze(0)


def infer(image_path, ckpt_path, device, save_dir='figures/classifier_torch/'):
    """
    Runs inference on an input image using a ResNet model.

    Args:
        image_path: The path to the input image
        ckpt_path: The path to the checkpoint file for the ResNet model
        device: The device to run inference on (CPU or GPU)
        save_dir: Inference result saving dir

    Returns:
        None
    """
    create_folder(save_dir)

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

    # post-process and get output
    classes = {0: 'normal', 1: 'pneumonia'}
    _, out = torch.max(output, dim=1)
    label = classes[out.item()]
    print(f"Result: {label}")

    # save result
    img_array = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    resized_src = cv.resize(img_array, None, fx=10, fy=10)
    cv.putText(resized_src, "Prediction:" + str(label), (0, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    file_name = image_path.split('/')[-1]
    cv.imwrite(save_dir + file_name, resized_src)


if __name__ == '__main__':
    # Define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True, default='pretrained/resnet18-sam.pth',
                        help='checkpoints path for inference')
    parser.add_argument('--image-path', type=str, required=True, default='imgs/pneumonia_img2.png',
                        help='image path for inference')
    parser.add_argument('--device', type=str, default='cpu',
                        help='inference device. [cuda | cpu]')
    args = parser.parse_args()

    # Check input arguments
    if not os.path.exists(args.ckpt_path):
        print(f'Cannot find the cnn checkpoints: {args.ckpt_path}')
        exit()
    if not os.path.exists(args.image_path):
        print(f'Cannot find the input image: {args.image_path}')
        exit()

    # Run to inference
    infer(args.image_path, args.ckpt_path, args.device)
