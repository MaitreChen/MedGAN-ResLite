from PIL import Image
import numpy as np
import cv2
import os

import torchvision.transforms as transforms
import torch.nn as nn
import torch

from models.model import create_model

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.00924097], std=[0.00282327])
])


def returnCAM(feature_conv, weight_softmax, class_idx):
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


def generate(img_path, CAM_RESULT_PATH='data/heatmap/normal/'):
    if not os.path.exists(CAM_RESULT_PATH):
        os.makedirs(CAM_RESULT_PATH)

    class_ = {0: 'normal', 1: 'pneumonia'}

    # Process img
    _, img_name = os.path.split(img_path)
    img = Image.open(img_path).convert('L')
    img_tensor = data_transform(img).unsqueeze(0)

    features = model_features(img_tensor).detach().cpu().numpy()

    outs = model_ft(img_tensor)
    res = torch.nn.functional.softmax(outs, dim=1).data.squeeze()
    _, idx = res.sort(0, True)
    idx = idx.cpu().numpy()

    CAMs = returnCAM(features, fc_weights, idx)

    img = cv2.imread(img_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)),
                                cv2.COLORMAP_JET)  # CAM resize match input image size
    fusion = cv2.addWeighted(heatmap, 0.5, img, 0.7, gamma=0.5)

    image_name_ = img_name.split(".")[-2]
    save_path = CAM_RESULT_PATH + image_name_ + '___' + class_[idx[0]] + '.png'

    cv2.imwrite(save_path, fusion)


if __name__ == '__main__':
    img_path = 'data/real/test/normal/img_3897.png'
    ckpt_path = 'checkpoints/cls/exp_ratio_1d1_resnet18_sam_meanAstd_lr_0.001_cos_20/cls_best.pth'
    save_dir = '/'.join(ckpt_path.split('/')[:-1])

    # Build model
    model = create_model('resnet18-sam', pretrained=False)
    if model is None:
        assert "model can not be None!"
    model_ft = model.cpu()
    model_ft.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

    # Get last feature map
    model_features = nn.Sequential(*list(model_ft.children())[:-2])
    fc_weights = model_ft.state_dict()['fc.weight'].cpu().numpy()

    model_ft.eval()
    model_features.eval()

    # Run and save
    test_files = os.listdir('data/real/test/normal')
    for test_file_name in test_files:
        test_img_path = os.path.join('data/real/test/normal', test_file_name)
        generate(ckpt_path, test_img_path)

    # for single img
    # generate(ckpt_path, img_path)
