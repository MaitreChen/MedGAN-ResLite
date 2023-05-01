from PIL import Image
import numpy as np
import argparse
import cv2
import os

from torch.nn import functional as F
import torch.nn as nn
import torch

from models.create_model import create_model
from get_data_info import get_transform


def build_model(ckpt_path):
    """
    This function builds a model from a given checkpoint path and returns the model, last feature map, and fully connected layer weights.

    Args:
    - ckpt_path: the path to the checkpoint file

    Returns:
    - model_ft: the built model
    - model_features: the last feature map of the model
    - fc_weights: the weights of the fully connected layer of the model
    """
    model = create_model('resnet18-sam', pretrained=False)
    if model is None:
        assert "model can not be None!"
    model_ft = model.cpu()
    model_ft.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

    # Get last feature map
    model_features = nn.Sequential(*list(model_ft.children())[:-2])
    fc_weights = model_ft.state_dict()['fc.weight'].cpu().numpy()

    return model_ft, model_features, fc_weights


def return_cam(feature_conv, weight_softmax, class_idx):
    """
    This function generates a heatmap for a given image and class index.

    Args:
    - feature_conv: the last convolutional layer of the model
    - weight_softmax: the weight of the fully connected layer of the model
    - class_idx: the index of the class to generate the heatmap for

    Returns:
    - output_cam: the heatmap for the given image and class index
    """
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        feature_conv = feature_conv.reshape((nc, h * w))
        cam = weight_softmax[idx].dot(
            feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cam_img)

    return output_cam


def generate_cam(img_path, ckpt_path, save_dir='figures/heatmap/'):
    """
    This function generates a heatmap for a given image and saves it to a specified directory.

    Args:
    - img_path: the path to the image to generate the heatmap for
    - SAVE_DIR: the directory to save the heatmap image to

    Returns:
    - None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    classes = {0: 'normal', 1: 'pneumonia'}

    # load img and pre-process
    _, img_name = os.path.split(img_path)
    img = Image.open(img_path).convert('L')
    img_tensor = get_transform()(img).unsqueeze(0)

    # build model
    model_ft, model_features, fc_weights = build_model(ckpt_path)
    model_ft.eval()
    model_features.eval()

    # get last feature map and results
    features = model_features(img_tensor).detach().cpu().numpy()

    outs = model_ft(img_tensor)
    h_x = F.softmax(outs, dim=1).data.squeeze()
    _, idx = h_x.sort(0, True)
    idx = idx.cpu().numpy()

    # generate class activation mapping for the top1 prediction
    CAMs = return_cam(features, fc_weights, idx)

    # render the CAM and output
    print(f'prediction: {classes[idx[0]]}')
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    fusion = cv2.addWeighted(heatmap, 0.5, img, 0.7, gamma=0.5)

    # save result
    image_name = img_name.split(".")[-2]
    save_path = save_dir + image_name + '_' + classes[idx[0]] + '.png'
    cv2.imwrite(save_path, fusion)


if __name__ == '__main__':
    # Define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, default='pretrained/resnet18-sam.pth',
                        help='checkpoints path for inference')
    parser.add_argument('--image-path', type=str, required=True, default='imgs/pneumonia_img2.png',
                        help='image path for cam')
    args = parser.parse_args()

    # Check input arguments
    if not os.path.exists(args.ckpt_path):
        print(f'Cannot find the checkpoints: {args.ckpt_path}')
        exit()
    if not os.path.exists(args.image_path):
        print(f'Cannot find the input image: {args.image_path}')
        exit()

    # Generate for single image
    generate_cam(args.image_path, args.ckpt_path)

    # Run and save
    # test_files = os.listdir('../data/real/test/normal')
    # for test_file_name in test_files:
    #     test_img_path = os.path.join('../data/real/test/normal', test_file_name)
    #     generate(ckpt_path, test_img_path)
