"""
Here is a script for sn-dcgan to generate image visualizations.
At the same time, you can use this script to generate minority class samples to train a classifier.
"""
from time import time
import cv2 as cv
import argparse
import os

import torchvision
import torch

from utils.image_utils import decode_to_numpy
from utils.common_utils import create_folder

# from models.gan import Generator
# from models.dcgan import Generator


from models.sndcgan import Generator


def infer(ckpt_path, batch_size, z_dim, image_size, device, save_path, save_mode, save_dir='figures/generator_torch/'):
    """
    Args:
        ckpt_path: checkpoints of dcgan
        batch_size: #
        z_dim: default 128
        image_size: default 128
        device: #
        save_path: #
        save_mode: 0: generate images for training; 1:save mini-batch images(Sprite map); -1: generate single image
        save_dir: Inference result saving dir
    """
    create_folder(save_dir)

    out_channels = 1

    # Create random tensors
    fixed_z = torch.randn(batch_size, z_dim)
    fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)
    # print(f"fixed_z.shape={fixed_z.shape}")

    # Define model and load checkpoint
    G = Generator(z_dim=z_dim, image_size=image_size, out_channels=out_channels)
    G.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda')))
    G.to(device)
    G.eval()

    # Start Inference
    start_time = time()
    fake_images = G(fixed_z.to(device))
    end_time = time()
    # print(f"Infer time: {end_time - start_time:.6f} ms")

    # Generate & Save images
    if save_mode == -1:
        print("Generate a single image...")
        fake_image_numpy, fake_image = decode_to_numpy(fake_images[0])
        fake_image = fake_image.transpose(1, 2, 0)
        cv.imwrite(save_dir + 'fake.png', fake_image)
    elif save_mode == 1:
        print("Generate a Sprite map...")
        create_folder(save_path + 'single/')
        idx = 0
        for i in range(batch_size):
            fake_image_numpy, fake_image = decode_to_numpy(fake_images[i][0])
            path = os.path.join(save_path, 'single', f'fake_img{idx}.png')
            cv.imwrite(path, fake_image)
            idx += 1
    elif save_mode == 0:
        print("Generate a batch of images...")
        create_folder(save_path + 'sprite/')
        save_fake_path = f"{save_path}/sprite/fake.png"
        torchvision.utils.save_image(fake_images, save_fake_path)


if __name__ == "__main__":
    # Define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True, default='pretrained/sn-dcgan.pth',
                        help='sn-dcgan checkpoints path for inference')
    parser.add_argument('--batch-size', type=int, required=True, default=1,
                        help='batch size for inference.')
    parser.add_argument('--image-size', type=int, default=64,
                        help='image size (default: 128)')
    parser.add_argument('--z-dim', type=int, default=128,
                        help='z-dim of generator (default: 128)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='inference device. [cuda | cpu]')
    parser.add_argument('--save-path', type=str, default='outs/',
                        help=' the path to save the generation fake images (single & sprite) (default: outs/)')
    parser.add_argument('--mode', type=int, default=-1, required=True,
                        help='generate images for [single:-1 | images:0 | sprite:1] (default: -1)')
    args = parser.parse_args()

    # Check input argument
    if not os.path.exists(args.ckpt_path):
        print(f'Cannot find the dcgan checkpoints: {args.ckpt_path}')
        exit()

    # Run to generate fake images
    infer(args.ckpt_path, args.batch_size, args.z_dim, args.image_size, args.device, args.save_path, args.mode)
